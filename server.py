# server.py
import os
import argparse
import flwr as fl
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import Net
from collections import defaultdict

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Federated Learning Server")
parser.add_argument("--rounds", type=int, default=10, help="Number of federated learning rounds")
parser.add_argument("--min-clients", type=int, default=1, help="Minimum number of clients to start a round")
parser.add_argument("--port", type=int, default=8080, help="Server port")
args = parser.parse_args()

# Create directory for results
os.makedirs("results", exist_ok=True)

# Global variables for tracking metrics
global_loss = []
global_acc = []
weights_over_time = []
client_contributions = defaultdict(list)
client_status = {}  # Track client status (active/failed)

# Evaluate global model on test dataset
def evaluate_global_model(parameters):
    """Evaluate the global model on the MNIST test dataset"""
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = Net().to(device)
    
    # Load parameters into model
    state_dict = {}
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    
    # Load test dataset
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    
    # Evaluate model
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += loss_fn(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    avg_loss = loss / len(testloader)
    
    return avg_loss, accuracy

# Custom Federated Averaging Strategy
class FedAvgWithFailures(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_first_round = {}  # Track when each client first joined
        self.client_last_round = {}   # Track the last round each client participated
        self.client_properties = {}   # Track client properties
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate client results and handle failures"""
        if not results:
            print(f"⚠️ Round {server_round} - No clients returned results")
            return None, {}
        
        if failures:
            print(f"⚠️ Round {server_round} - {len(failures)} clients failed")
            for failure in failures:
                if isinstance(failure, tuple) and len(failure) > 0:
                    client_id = failure[0].cid
                    client_status[client_id] = "failed"
                    print(f"Client {client_id} failed")
        
        # Process client results with weighted aggregation
        weighted_results = []
        total_weight = 0.0
        
        for client, fit_res in results:
            client_id = client.cid
            client_status[client_id] = "active"
            
            # Track first participation
            if client_id not in self.client_first_round:
                self.client_first_round[client_id] = server_round
                print(f"New client joined: {client_id}")
            
            # Calculate missed rounds
            missed_rounds = 0
            if client_id in self.client_last_round:
                missed_rounds = server_round - self.client_last_round[client_id] - 1
            
            # Update last active round
            self.client_last_round[client_id] = server_round
            
            # Store client information
            dataset_size = fit_res.num_examples
            self.client_properties[client_id] = {"dataset_size": dataset_size}
            
            # Calculate weight based on dataset size and missed rounds
            weight = dataset_size
            if missed_rounds > 0:
                # Apply penalty for missed rounds: exp(-missed/5)
                missed_round_penalty = np.exp(-missed_rounds / 5)
                weight = weight * missed_round_penalty
                print(f"Client {client_id}: {dataset_size} samples, missed {missed_rounds} rounds, weight={weight:.2f}")
            else:
                print(f"Client {client_id}: {dataset_size} samples, no missed rounds, weight={weight:.2f}")
            
            total_weight += weight
            weighted_results.append((client, fit_res, weight))
            
            # Track contribution for visualization
            parameters = fl.common.parameters_to_ndarrays(fit_res.parameters)
            norm = np.linalg.norm(np.concatenate([p.flatten() for p in parameters]))
            
            # Fill missing rounds with zeros
            while len(client_contributions[client_id]) < server_round - 1:
                client_contributions[client_id].append(0.0)
            
            # Add current contribution
            client_contributions[client_id].append(norm)
        
        # Normalize weights
        if total_weight > 0:
            weighted_results = [(client, fit_res, weight / total_weight) 
                               for client, fit_res, weight in weighted_results]
        
        # Aggregate parameters
        parameters_aggregated = self._aggregate_fit_params(weighted_results)
        
        if parameters_aggregated is not None:
            # Evaluate global model
            parameters_list = fl.common.parameters_to_ndarrays(parameters_aggregated)
            loss, acc = evaluate_global_model(parameters_list)
            print(f"📊 Round {server_round} - Global model evaluation: Loss={loss:.4f}, Accuracy={acc:.4f}")
            
            # Record metrics for visualization
            global_loss.append(loss)
            global_acc.append(acc)
            
            # Track weight evolution
            if len(parameters_list) > 0:
                mean_weight = np.mean(parameters_list[0])
                weights_over_time.append(mean_weight)
            
            # Fill in missing contribution data for visualization
            for cid in client_contributions:
                while len(client_contributions[cid]) < server_round:
                    client_contributions[cid].append(0.0)
        
        return parameters_aggregated, {}
    
    def _aggregate_fit_params(self, results):
        """Aggregate parameters using weighted average"""
        weights_results = [
            (fl.common.parameters_to_ndarrays(fit_res.parameters), weight)
            for _, fit_res, weight in results
        ]
        
        return fl.common.ndarrays_to_parameters(
            fl.common.aggregate.weighted_average(weights_results)
        )

def create_visualizations():
    """Generate and save visualization plots"""
    print("Creating visualizations...")
    
    # 1. Global model loss over rounds
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(global_loss) + 1), global_loss, marker='o')
    plt.title("Global Model Loss over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("results/global_loss.png")
    plt.close()
    
    # 2. Global model accuracy over rounds
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(global_acc) + 1), global_acc, marker='o')
    plt.title("Global Model Accuracy over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("results/global_accuracy.png")
    plt.close()
    
    # 3. Weight evolution
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(weights_over_time) + 1), weights_over_time, marker='o')
    plt.title("Weight Evolution in Federated Learning")
    plt.xlabel("Round")
    plt.ylabel("Mean Weight Value")
    plt.grid(True)
    plt.savefig("results/weight_evolution.png")
    plt.close()
    
    # 4. Client contributions
    plt.figure(figsize=(12, 7))
    for client_id, contributions in client_contributions.items():
        rounds = range(1, len(contributions) + 1)
        plt.plot(rounds, contributions, marker='o', label=f"Client {client_id[:6]}")
        
        # Mark failures with red X
        failed_rounds = [i+1 for i, c in enumerate(contributions) if c == 0]
        if failed_rounds:
            plt.scatter(failed_rounds, [0] * len(failed_rounds), marker='x', color='red', s=100)
    
    plt.title("Client Contributions Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Contribution (Parameter Norm)")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/client_contributions.png")
    plt.close()
    
    print("✅ Visualizations saved to 'results' directory")

def main():
    """Start the federated learning server"""
    # Create a local model to provide initial parameters
    initial_model = Net()
    initial_parameters = [param.cpu().detach().numpy() for param in initial_model.parameters()]
    
    # Initialize custom strategy with initial parameters
    strategy = FedAvgWithFailures(
        fraction_fit=1.0,  # Use all available clients
        min_fit_clients=args.min_clients,
        min_available_clients=args.min_clients,
        min_evaluate_clients=0,  # Skip evaluation for simplicity
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),  # Add initial parameters
    )
    
    # Server configuration
    server_config = fl.server.ServerConfig(num_rounds=args.rounds)
    
    print(f"🚀 Starting Federated Learning server on port {args.port}")
    print(f"Running for {args.rounds} rounds with minimum {args.min_clients} clients")
    
    # Start server
    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=server_config,
        strategy=strategy,
    )
    
    # Create visualizations after training
    create_visualizations()

if __name__ == "__main__":
    main()