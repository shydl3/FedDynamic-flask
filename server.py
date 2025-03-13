import flwr as fl
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from model import Net

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
    
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total
    return 0.0, accuracy  # Dummy loss, actual accuracy

# Custom aggregation strategy
class FedAvgWithFailureHandling(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_missed_rounds = 3
        self.contribution_threshold = 0.7
    
    def aggregate_fit(self, server_round, results, failures):
        global_metrics["rounds"] = server_round
        
        if not results:
            return None, {}
        
        # Process client contributions
        processed_results = []
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            
            # Check for client metrics
            if fit_res.metrics and "status" in fit_res.metrics:
                status = fit_res.metrics["status"]
                
                # Process active clients
                if status == "active":
                    # Update client record
                    if client_id not in global_metrics["client_status"]:
                        global_metrics["client_status"][client_id] = {
                            "active": True,
                            "rounds_participated": 0,
                            "rounds_missed": 0
                        }
                    
                    client_status = global_metrics["client_status"][client_id]
                    client_status["active"] = True
                    client_status["rounds_participated"] += 1
                    client_status["rounds_missed"] = 0
                    
                    # Save dataset size
                    if "dataset_size" in fit_res.metrics:
                        global_metrics["client_dataset_size"][client_id] = fit_res.metrics["dataset_size"]
                    
                    # Calculate contribution (using parameter norm)
                    parameters = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    norm = np.linalg.norm(np.concatenate([p.flatten() for p in parameters]))
                    
                    # Track contribution
                    global_metrics["client_contributions"][client_id].append(norm)
                    
                    # Keep this result for aggregation
                    processed_results.append((client_proxy, fit_res))
                    
                # Handle failed clients
                elif status == "failed":
                    if client_id not in global_metrics["client_status"]:
                        global_metrics["client_status"][client_id] = {
                            "active": False,
                            "rounds_participated": 0,
                            "rounds_missed": 1
                        }
                    else:
                        global_metrics["client_status"][client_id]["active"] = False
                        global_metrics["client_status"][client_id]["rounds_missed"] += 1
                    
                    # Add zero contribution for visualization
                    global_metrics["client_contributions"][client_id].append(0)
                    
                    print(f"⚠️ Client {client_id[:8]} failed in round {server_round}")
        
        # Normalize contribution lengths
        max_contribs = max([len(contribs) for contribs in global_metrics["client_contributions"].values()], default=0)
        for client_id in global_metrics["client_contributions"]:
            while len(global_metrics["client_contributions"][client_id]) < max_contribs:
                global_metrics["client_contributions"][client_id].append(0)
        
        # Handle cases with no valid results
        if not processed_results:
            print(f"❌ Round {server_round}: No valid client updates")
            return None, {}
        
        # Use parent class for aggregation
        parameters_aggregated, metrics = super().aggregate_fit(server_round, processed_results, failures)
        
        if parameters_aggregated is not None:
            # Evaluate global model
            ndarrays = fl.common.parameters_to_ndarrays(parameters_aggregated)
            loss, accuracy = evaluate_global_model(ndarrays)
            
            # Store metrics
            global_metrics["loss"].append(loss)
            global_metrics["accuracy"].append(accuracy)
            
            # Track weight evolution
            if len(ndarrays) > 0:
                global_metrics["weights_evolution"].append(float(np.mean(ndarrays[0])))
            
            print(f"📊 Round {server_round}: Accuracy={accuracy:.4f}")
            
            # Update metrics
            metrics["accuracy"] = accuracy
        
        return parameters_aggregated, metrics
    
    def configure_fit(
        self, server_round: int, parameters, client_manager
    ):
        """Configure clients for training - handle rejoining clients"""
        
        # Get all clients and their configuration
        config = {}
        
        # Get all available client IDs
        available_clients = [client.cid for client in client_manager.all()]
        
        # Check if any clients rejoined after failure
        for client_id, status in global_metrics["client_status"].items():
            if status["active"] == False and client_id in available_clients:
                # Client is rejoining
                missed_rounds = status["missed_rounds"]
                
                # Decide if client can keep weights or not
                if missed_rounds <= self.max_missed_rounds:
                    # Client can keep its weights
                    print(f"🔄 Client {client_id[:8]} rejoined after {missed_rounds} rounds - keeping weights")
                    config[client_id] = {"keep_weights": True}
                else:
                    # Client gets fresh weights
                    print(f"🆕 Client {client_id[:8]} rejoined after {missed_rounds} rounds - resetting weights")
                    config[client_id] = {"keep_weights": False}
                
                # Mark as active again
                status["active"] = True
        
        return super().configure_fit(server_round, parameters, client_manager)

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