# client.py
import argparse
import random
import os
import json
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import flwr as fl
import numpy as np
from model import Net

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Federated Learning Client")
parser.add_argument("--id", type=str, help="Client ID")
parser.add_argument("--dataset-size", type=int, help="Size of dataset for this client")
parser.add_argument("--failure-prob", type=float, default=0.1, help="Probability of failure per round")
parser.add_argument("--recovery-prob", type=float, default=0.7, help="Probability of recovery per round")
parser.add_argument("--server-address", type=str, default="127.0.0.1:8080", help="Server address")
args = parser.parse_args()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup client ID and state directory
os.makedirs("client_states", exist_ok=True)
client_id = args.id if args.id else str(uuid.uuid4())[:8]
state_file = f"client_states/client_{client_id}.json"

# Load or initialize client state
if os.path.exists(state_file):
    with open(state_file, 'r') as f:
        client_state = json.load(f)
    print(f"Client {client_id}: Loaded existing state")
else:
    client_state = {
        "id": client_id,
        "rounds_participated": 0,
        "last_round": 0,
        "is_failed": False,
        "failed_rounds": []
    }
    with open(state_file, 'w') as f:
        json.dump(client_state, f)
    print(f"Client {client_id}: Initialized new state")

# Load MNIST dataset and partition based on client ID
def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    
    # Determine dataset size
    if args.dataset_size:
        dataset_size = min(args.dataset_size, len(mnist_train))
    else:
        # Default dataset sizes based on client ID
        if "1" in client_id:
            dataset_size = 10000  # Client 1: 10k samples
        elif "2" in client_id:
            dataset_size = 20000  # Client 2: 20k samples
        else:
            dataset_size = 30000  # Client 3: 30k samples
    
    # Create client dataset
    indices = torch.randperm(len(mnist_train))[:dataset_size]
    dataset = torch.utils.data.Subset(mnist_train, indices)
    
    print(f"Client {client_id}: Loaded {len(dataset)} samples")
    return dataset

# Train function
def train(model, train_loader, epochs=1):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return total_loss / len(train_loader)

# Federated Learning Client
class MNISTClient(fl.client.NumPyClient):
    def __init__(self, client_id, failure_prob, recovery_prob):
        self.model = Net().to(device)
        self.client_id = client_id
        self.failure_prob = failure_prob
        self.recovery_prob = recovery_prob
        self.dataset = load_data()
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=32, shuffle=True
        )
        self.weights_file = f"client_states/weights_{client_id}.pt"
        
        # Load existing weights if available
        if os.path.exists(self.weights_file):
            self.model.load_state_dict(torch.load(self.weights_file))
            print(f"Client {client_id}: Loaded existing weights")
    
    def get_parameters(self, config):
        """Return model parameters as numpy array list"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters from numpy array list"""
        state_dict = {}
        idx = 0
        for key in self.model.state_dict().keys():
            state_dict[key] = torch.tensor(parameters[idx], device=device)
            idx += 1
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Train the model on local dataset"""
        global client_state
        
        # Update current round
        current_round = int(config.get("round", 0))
        if current_round == 0:  # If round info not provided, increment
            current_round = client_state["rounds_participated"] + 1
        
        # Simulate failure
        if not client_state["is_failed"] and random.random() < self.failure_prob:
            client_state["is_failed"] = True
            client_state["failed_rounds"].append(current_round)
            print(f"⚠️ Client {self.client_id} failed at round {current_round}")
            
            # Save state and weights
            with open(state_file, 'w') as f:
                json.dump(client_state, f)
            torch.save(self.model.state_dict(), self.weights_file)
            
            return None, 0, {}
        
        # Handle recovery
        if client_state["is_failed"]:
            if random.random() < self.recovery_prob:
                client_state["is_failed"] = False
                print(f"✅ Client {self.client_id} recovered at round {current_round}")
            else:
                # Still failed
                client_state["failed_rounds"].append(current_round)
                print(f"⚠️ Client {self.client_id} still failed at round {current_round}")
                
                # Save state
                with open(state_file, 'w') as f:
                    json.dump(client_state, f)
                return None, 0, {}
        
        # Calculate missed rounds
        missed_rounds = 0
        if client_state["last_round"] > 0:
            missed_rounds = current_round - client_state["last_round"] - 1
        
        # Decide whether to reset weights
        if missed_rounds > 0:
            # Formula: if missed_rounds > dataset_size/5000, reset weights
            dataset_weight_factor = len(self.dataset) / 5000
            if missed_rounds > dataset_weight_factor:
                print(f"🔄 Client {self.client_id} missed {missed_rounds} rounds - resetting weights")
                # Set received parameters
                self.set_parameters(parameters)
            else:
                print(f"📈 Client {self.client_id} missed {missed_rounds} rounds but keeps its weights")
                # Keep current weights
        else:
            # Set received parameters
            self.set_parameters(parameters)
        
        # Train model
        loss = train(self.model, self.train_loader)
        
        # Update client state
        client_state["rounds_participated"] += 1
        client_state["last_round"] = current_round
        
        # Save state and weights
        with open(state_file, 'w') as f:
            json.dump(client_state, f)
        torch.save(self.model.state_dict(), self.weights_file)
        
        print(f"📊 Client {self.client_id} completed round {current_round}, Loss: {loss:.4f}")
        
        # Return updated parameters, dataset size, and metrics
        updated_params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return updated_params, len(self.dataset), {"loss": loss}
    
    def evaluate(self, parameters, config):
        """Evaluate the model on local dataset"""
        self.set_parameters(parameters)
        self.model.eval()
        
        loss = 0.0
        total = 0
        correct = 0
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.train_loader:
                data, target = data.to(device), target.to(device)
                outputs = self.model(data)
                loss += criterion(outputs, target).item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        avg_loss = loss / len(self.train_loader)
        
        return float(avg_loss), len(self.dataset), {"accuracy": accuracy}

# Main function
def main():
    # Create client instance
    client = MNISTClient(
        client_id=client_id,
        failure_prob=args.failure_prob,
        recovery_prob=args.recovery_prob
    )
    
    print(f"🚀 Client {client_id} starting with:")
    print(f"   - Dataset size: {len(client.dataset)} samples")
    print(f"   - Failure probability: {args.failure_prob}")
    print(f"   - Recovery probability: {args.recovery_prob}")
    print(f"   - Server address: {args.server_address}")
    
    # Start client
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

if __name__ == "__main__":
    main()