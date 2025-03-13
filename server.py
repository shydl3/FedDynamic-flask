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
import matplotlib.patches as mpatches

# Create results directory
os.makedirs("results", exist_ok=True)

# Global tracking variables
global_metrics = {
    "rounds": 0,
    "loss": [],
    "accuracy": [],
    "weights_evolution": [],
    "client_reliability": defaultdict(list),  # Changed from contributions to reliability
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
        global_metrics["rounds"] = server_round
        
        # Process results for global metrics
        if results:
            # Convert parameters for evaluation
            parameters_aggregated, _ = super().aggregate_fit(server_round, results, [])
            if parameters_aggregated is not None:
                ndarrays = fl.common.parameters_to_ndarrays(parameters_aggregated)
                
                # Evaluate global model
                loss, accuracy = evaluate_global_model(ndarrays)
                
                # Store metrics
                global_metrics["loss"].append(loss)
                global_metrics["accuracy"].append(accuracy)
                
                # Store weight evolution (mean of first layer)
                if len(ndarrays) > 0:
                    global_metrics["weights_evolution"].append(float(np.mean(ndarrays[0])))
                
                print(f"📊 Round {server_round}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        
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
            
            # Calculate reliability score
            reliability_score, _ = self.calculate_client_reliability_score(client_id, server_round)
            
            # Update global metrics for visualization
            global_metrics["client_reliability"][client_id].append({
                "round": server_round,
                "reliability": reliability_score,
                "status": "success",
                "data_recovered": False  # Not applicable for success
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
            
            # Calculate reliability score for the failed client
            reliability_score, _ = self.calculate_client_reliability_score(client_id, server_round)
            
            # Determine if data can be recovered based on reliability
            data_recovered = reliability_score > 0.7  # Example threshold
            
            # Update global metrics for visualization
            global_metrics["client_reliability"][client_id].append({
                "round": server_round,
                "reliability": reliability_score,
                "status": "failure",
                "data_recovered": data_recovered
            })
        
        # Continue with original aggregation logic
        return super().aggregate_fit(server_round, results, failures)
    
    def configure_fit(self, server_round, parameters, client_manager):
        """Configure clients for training - handle rejoining clients"""
        self.current_round = server_round
        
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
        """Calculate client reliability score using exponential time decay."""
        # Get client history or initialize empty if new client
        history = self.client_history.get(client_id, {"participation_records": []})
        participation_records = history.get("participation_records", [])
        
        if not participation_records:
            # For new clients, initialize their record and return default score
            self.client_history[client_id] = {"participation_records": []}
            return 0.5, False  # Default moderate reliability for new clients
        
        # Constants
        decay_rate = 0.2  # Increased for more responsiveness
        max_history = 20  # Maximum rounds to consider
        
        # Initialize score components
        reliability_score = 0.0
        normalization_factor = 0.0
        
        # Process each historical record with exponential time decay
        for record in sorted(participation_records[-max_history:], key=lambda x: x["round"]):
            # Calculate time decay factor (more recent = more important)
            time_diff = current_round - record["round"]
            time_weight = math.exp(-decay_rate * time_diff)
            
            # Calculate impact based on record type - FIXED VALUES HERE
            if record["status"] == "success":
                # Successful participation: scaled down to reasonable values
                contribution = record.get("contribution", 0.1)
                impact = 0.3 * (1.0 + min(1.0, contribution * 2.0))
            elif record["status"] == "failure":
                # Failed during training: increased negative impact
                impact = -1.0
            else:  # "missed"
                impact = -0.5
            
            # Add weighted impact to score
            reliability_score += impact * time_weight
            normalization_factor += time_weight
        
        # Normalize to 0-1 range - FIXED CALCULATION HERE
        if normalization_factor > 0:
            # Create a more balanced range by using larger denominator
            reliability_score = 0.5 + (reliability_score / (2.5 * normalization_factor))
            reliability_score = max(0.1, min(0.95, reliability_score))  # Avoid extreme values
        else:
            reliability_score = 0.5
        
        # Debug output if needed
        if self.verbose:
            print(f"Client {client_id[:8]}: reliability={reliability_score:.3f}")
        
        # Determine whether to keep weights
        last_seen = max([r["round"] for r in participation_records]) if participation_records else 0
        rounds_missed = current_round - last_seen
        
        # Adaptive threshold based on absence duration
        threshold = 0.6 * math.exp(-0.05 * rounds_missed)
        keep_weights = reliability_score >= threshold
        
        return reliability_score, keep_weights

# Function to create visualizations
def create_visualizations():
    print(f"Creating visualizations with {len(global_metrics['loss'])} loss points, {len(global_metrics['accuracy'])} accuracy points")
    
    # 1. Plot global loss
    plt.figure(figsize=(10, 6))
    if global_metrics["loss"]:
        plt.plot(range(1, len(global_metrics["loss"])+1), global_metrics["loss"], marker='o')
    else:
        plt.text(0.5, 0.5, "No loss data available", horizontalalignment='center')
    plt.title("Global Model Loss over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("results/global_loss.png")
    plt.close()
    
    # 2. Plot global accuracy
    plt.figure(figsize=(10, 6))
    if global_metrics["accuracy"]:
        plt.plot(range(1, len(global_metrics["accuracy"])+1), global_metrics["accuracy"], marker='o')
    else:
        plt.text(0.5, 0.5, "No accuracy data available", horizontalalignment='center')
    plt.title("Global Model Accuracy over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("results/global_accuracy.png")
    plt.close()
    
    # 3. Plot weight evolution
    plt.figure(figsize=(10, 6))
    if global_metrics["weights_evolution"]:
        plt.plot(range(1, len(global_metrics["weights_evolution"])+1), global_metrics["weights_evolution"], marker='o')
    else:
        plt.text(0.5, 0.5, "No weight evolution data available", horizontalalignment='center')
    plt.title("Weight Evolution in Federated Learning")
    plt.xlabel("Round")
    plt.ylabel("Mean Weight Value")
    plt.grid(True)
    plt.savefig("results/weights_evolution.png")
    plt.close()
    
    # 4. Plot client reliability (replacing client contributions)
    plt.figure(figsize=(12, 7))
    
    # Define markers
    success_marker = 'o'  # Circle for normal operation
    failure_marker = 'x'  # X for failure
    recovery_marker = '+'  # Plus for recovered data
    
    # Track all rounds for x-axis
    all_rounds = set()
    
    # Plot each client's reliability
    for client_id, reliability_data in global_metrics["client_reliability"].items():
        # Sort data by round
        reliability_data.sort(key=lambda x: x["round"])
        
        # Extract rounds and reliability scores
        rounds = [entry["round"] for entry in reliability_data]
        reliability_scores = [entry["reliability"] for entry in reliability_data]
        
        # Update all rounds
        all_rounds.update(rounds)
        
        # Plot the reliability line
        plt.plot(rounds, reliability_scores, '-', label=f"Client {client_id[:8]}")
        
        # Plot points with appropriate markers
        for entry in reliability_data:
            if entry["status"] == "failure":
                if entry["data_recovered"]:
                    # Failure with data recovery
                    plt.scatter(entry["round"], entry["reliability"], 
                              marker=recovery_marker, s=100, color='green', zorder=5)
                else:
                    # Failure without data recovery
                    plt.scatter(entry["round"], entry["reliability"], 
                              marker=failure_marker, s=100, color='red', zorder=5)
            else:
                # Normal operation
                plt.scatter(entry["round"], entry["reliability"], 
                          marker=success_marker, s=80, zorder=3)
    
    # Customize plot appearance
    plt.title("Client Reliability Scores Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Reliability Score")
    plt.ylim(0, 1.1)  # Reliability scores are between 0 and 1
    plt.grid(True)
    
    # Create legend for client lines
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Create legend for markers
    success_patch = mpatches.Patch(color='blue', label='Success (o)')
    failure_patch = mpatches.Patch(color='red', label='Failure (x)')
    recovery_patch = mpatches.Patch(color='green', label='Data Recovered (+)')
    
    # Add both sets of legend items
    all_handles = handles + [success_patch, failure_patch, recovery_patch]
    all_labels = labels + ['Success', 'Failure', 'Data Recovered']
    plt.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    plt.tight_layout()
    plt.savefig("results/client_reliability.png", bbox_inches='tight')
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