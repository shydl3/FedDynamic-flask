import flwr as fl
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from model import Net
import math

# Create results directory
os.makedirs("results", exist_ok=True)

# Global tracking variables
global_metrics = {
    "rounds": 0,
    "loss": [],
    "accuracy": [],
    "weights_evolution": [],
    "client_contributions": defaultdict(list),
    "client_status": {},
    "client_dataset_size": {}
}

# Function to evaluate global model
def evaluate_global_model(parameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    
    # Load parameters into model
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=False)
    
    # Load test data
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    
    # Evaluate
    model.eval()
    correct, total = 0, 0
    loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss += criterion(outputs, target).item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    average_loss = loss / total
    accuracy = correct / total
    return average_loss, accuracy

# Custom aggregation strategy
class FedAvgWithFailureHandling(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize client history tracking
        self.client_history = {}  # Dictionary to track client participation
        self.current_round = 0
        self.verbose = True  # For logging reliability calculations
    
    def aggregate_fit(self, server_round, results, failures):
        # Record successful clients
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            
            # Update client history with success
            if client_id not in self.client_history:
                self.client_history[client_id] = {"participation_records": []}
                
            # Calculate contribution based on num_examples or other metric
            contribution = fit_res.num_examples / max(sum(r.num_examples for _, r in results), 1)
            
            # Add successful participation record
            self.client_history[client_id]["participation_records"].append({
                "round": server_round,
                "status": "success",
                "contribution": contribution,
            })
        
        # Record failed clients
        for client_proxy in failures:
            client_id = client_proxy.cid
            
            # Update client history with failure
            if client_id not in self.client_history:
                self.client_history[client_id] = {"participation_records": []}
                
            # Add failure record
            self.client_history[client_id]["participation_records"].append({
                "round": server_round,
                "status": "failure"
            })
        
        # Continue with original aggregation logic
        return super().aggregate_fit(server_round, results, failures)
    
    def configure_fit(self, server_round, parameters, client_manager):
        """Configure clients for training - handle rejoining clients"""
        
        # Get all clients and their configuration
        config = {}
        
        # Get available client IDs
        available_clients = client_manager.all()
        
        # Check if any clients rejoined after failure
        for client_id, status in global_metrics["client_status"].items():
            # Initialize status if needed
            if "active" not in status:
                status["active"] = True
            if "missed_rounds" not in status:
                status["missed_rounds"] = 0
            
            if status["active"] == False and client_id in available_clients:
                # Client is rejoining
                reliability_score, keep_weights = self.calculate_client_reliability_score(
                    client_id, server_round
                )
                
                if keep_weights:
                    print(f"🔄 Client {client_id[:8]} rejoined - reliability: {reliability_score:.2f} - keeping weights")
                    config[client_id] = {"keep_weights": True}
                else:
                    print(f"🆕 Client {client_id[:8]} rejoined - reliability: {reliability_score:.2f} - resetting weights")
                    config[client_id] = {"keep_weights": False}
                
                # Mark as active again
                status["active"] = True
        
        return super().configure_fit(server_round, parameters, client_manager)
    
    def calculate_client_reliability_score(self, client_id, current_round):
        """
        Calculate client reliability score using exponential time decay.
        
        Args:
            self: The server instance
            client_id: The client's identifier
            current_round: Current training round number
        """
        # Get client history or initialize empty if new client
        history = self.client_history.get(client_id, {"participation_records": []})
        participation_records = history.get("participation_records", [])
        
        if not participation_records:
            # For new clients, initialize their record and return default score
            self.client_history[client_id] = {"participation_records": []}
            return 0.5, False  # Default moderate reliability for new clients
        
        # Constants
        decay_rate = 0.1  # Controls time decay (higher = faster decay)
        max_history = 20  # Maximum rounds to consider
        
        # Initialize score components
        reliability_score = 0.0
        normalization_factor = 0.0
        
        # Process each historical record with exponential time decay
        for record in sorted(participation_records[-max_history:], key=lambda x: x["round"]):
            # Calculate time decay factor (more recent = more important)
            time_diff = current_round - record["round"]
            time_weight = math.exp(-decay_rate * time_diff)
            
            # Calculate impact based on record type
            if record["status"] == "success":
                # Successful participation: positive impact based on contribution
                impact = 1.0 * (1.0 + min(1.0, record.get("contribution", 0.1) * 5.0))
            elif record["status"] == "failure":
                # Failed during training: significant negative impact
                impact = -1.0
            else:  # "missed"
                # Missed round: moderate negative impact
                impact = -0.5
            
            # Add weighted impact to score
            reliability_score += impact * time_weight
            normalization_factor += time_weight
        
        # Normalize to 0-1 range
        if normalization_factor > 0:
            # Transform from potentially negative values to 0-1 range
            reliability_score = 0.5 + (reliability_score / (2 * normalization_factor))
            reliability_score = max(0.0, min(1.0, reliability_score))
        else:
            reliability_score = 0.5
        
        # Determine whether to keep weights
        last_seen = max([r["round"] for r in participation_records]) if participation_records else 0
        rounds_missed = current_round - last_seen
        
        # Adaptive threshold based on absence duration
        threshold = 0.6 * math.exp(-0.05 * rounds_missed)
        keep_weights = reliability_score >= threshold
        
        return reliability_score, keep_weights

# Function to create visualizations
def create_visualizations():
    # 1. Plot global loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(global_metrics["loss"])+1), global_metrics["loss"], marker='o')
    plt.title("Global Model Loss over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("results/global_loss.png")
    plt.close()
    
    # 2. Plot global accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(global_metrics["accuracy"])+1), global_metrics["accuracy"], marker='o')
    plt.title("Global Model Accuracy over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("results/global_accuracy.png")
    plt.close()
    
    # 3. Plot weight evolution
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(global_metrics["weights_evolution"])+1), global_metrics["weights_evolution"], marker='o')
    plt.title("Weight Evolution in Federated Learning")
    plt.xlabel("Round")
    plt.ylabel("Mean Weight Value")
    plt.grid(True)
    plt.savefig("results/weights_evolution.png")
    plt.close()
    
    # 4. Plot client contributions
    plt.figure(figsize=(12, 7))
    
    for client_id, contributions in global_metrics["client_contributions"].items():
        rounds = range(1, len(contributions) + 1)
        
        # Plot regular contributions
        plt.plot(rounds, contributions, label=f"Client {client_id[:8]}", marker='o')
        
        # Mark rounds where client failed
        failed_rounds = [i+1 for i, c in enumerate(contributions) if c == 0]
        if failed_rounds:
            plt.scatter(failed_rounds, [0] * len(failed_rounds), color='red', s=80, marker='x', label=f"Client {client_id[:8]} failed" if client_id == list(global_metrics["client_contributions"].keys())[0] else "")
    
    plt.title("Client Contribution Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Contribution (Parameter Norm)")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/client_contributions.png")
    plt.close()
    
    print("✅ All visualizations saved to 'results' directory")

# Main server function
def main():
    # Define strategy
    strategy = FedAvgWithFailureHandling(
        min_fit_clients=1,  # Proceed even with just one client
        min_available_clients=1,
        min_evaluate_clients=0
    )
    
    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )
    
    # Create visualizations when done
    create_visualizations()

if __name__ == "__main__":
    main()