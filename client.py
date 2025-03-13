import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import flwr as fl
import numpy as np
import random
import argparse
import uuid
import os
import json
from model import Net

# Parse arguments
parser = argparse.ArgumentParser(description="Federated Learning Client")
parser.add_argument("--client_id", type=str, default=None)
parser.add_argument("--dataset_size", type=int, default=10000)
parser.add_argument("--fail_prob", type=float, default=0.1)
parser.add_argument("--recovery_prob", type=float, default=0.8)
parser.add_argument("--server", type=str, default="127.0.0.1:8080")
args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate persistent client ID if not provided
if args.client_id is None:
    client_id_file = f"client_{os.getpid()}_id.json"
    if os.path.exists(client_id_file):
        with open(client_id_file, "r") as f:
            args.client_id = json.load(f)["id"]
    else:
        args.client_id = str(uuid.uuid4())
        with open(client_id_file, "w") as f:
            json.dump({"id": args.client_id}, f)

print(f"🛩️  Client {args.client_id[:8]} starting - Dataset: {args.dataset_size}, Fail prob: {args.fail_prob}")

# Track client state
client_state = {
    "active": True,
    "rounds_participated": 0,
    "rounds_missed": 0,
    "rounds_total": 0
}

# Load MNIST dataset (subset based on dataset_size)
def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    
    # Take a subset based on dataset_size
    indices = torch.randperm(len(trainset))[:args.dataset_size]
    subset = torch.utils.data.Subset(trainset, indices)
    
    return subset

# Train function
def train(model, train_loader):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Define FL Client class
class MNISTClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = Net().to(device)
        self.trainset = load_data()
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=32, shuffle=True
        )
    
    def get_parameters(self, config):
        return [param.cpu().detach().numpy() for param in self.model.parameters()]
    
    def set_parameters(self, parameters):
        for i, param in enumerate(self.model.parameters()):
            param.data = torch.from_numpy(parameters[i]).to(device)
    
    def fit(self, parameters, config):
        client_state["rounds_total"] += 1
        
        # Check if client should fail
        if random.random() < args.fail_prob and client_state["active"]:
            client_state["active"] = False
            client_state["rounds_missed"] = 0
            print(f"❌ Client {args.client_id[:8]} failed")
            # Return current parameters instead of None
            empty_params = self.get_parameters(config)
            return empty_params, 0, {"status": "failed", "client_id": args.client_id}
        
        # Check if client should recover
        if not client_state["active"] and random.random() < args.recovery_prob:
            client_state["active"] = True
            print(f"✅ Client {args.client_id[:8]} recovered after {client_state['rounds_missed']} rounds")
        
        # If still inactive, update state and return failure
        if not client_state["active"]:
            client_state["rounds_missed"] += 1
            print(f"📴 Client {args.client_id[:8]} still inactive (missed {client_state['rounds_missed']} rounds)")
            empty_params = self.get_parameters(config)
            return empty_params, 0, {"status": "failed", "client_id": args.client_id}
        
        # Active client - proceed with training
        client_state["rounds_participated"] += 1
        
        # Load parameters and train
        self.set_parameters(parameters)
        train(self.model, self.trainloader)
        
        # Return updated parameters
        updated_params = self.get_parameters(config)
        
        print(f"📊 Client {args.client_id[:8]} completed round {client_state['rounds_total']}")
        
        return updated_params, len(self.trainloader.dataset), {
            "status": "active",
            "client_id": args.client_id,
            "dataset_size": args.dataset_size,
            "rounds_participated": client_state["rounds_participated"]
        }
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        testset = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(testset, batch_size=32)
        
        loss = 0.0
        correct = 0
        total = 0
        
        self.model.eval()
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = correct / total
        return 0.0, total, {"accuracy": accuracy}

# Start client
if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address=args.server,
        client=MNISTClient()
    )