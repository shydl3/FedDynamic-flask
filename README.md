# FedDynamic: Federated Learning System with Failure Handling

## Overview
Course project for CS230 implementing a federated learning system with dynamic client participation.

This system implements a robust federated learning framework that can tolerate client failures and handle client reconnections intelligently. The implementation uses PyTorch for the machine learning components and Flower (flwr) for the federated learning orchestration.

## System Architecture

The system consists of three main components:

1. **Server**: Coordinates the federated learning process, aggregates client updates, and handles client failures/reconnections
2. **Clients**: Train local models on their datasets and communicate with the server
3. **Launcher**: Orchestrates the startup of the server and multiple clients on a single machine

### Components:
- `server.py`: Implements the federated server with custom aggregation logic
- `client.py`: Implements clients with failure simulation capabilities
- `model.py`: Defines the neural network architecture (CNN for MNIST)
- `launcher.py`: Utility script to start the server and multiple clients locally

## Failure Handling Logic

The system handles client failures through a sophisticated mechanism:

1. **Failure Detection**: When a client fails (simulated or real), the server identifies the missing client.
2. **Client Rejoining**: When a client reconnects after failure:
   - The server evaluates if the client should keep its current weights or receive the global model.
   - This decision is based on:
     - Number of rounds missed (`missed_rounds`)
     - Client's previous contribution significance
   
3. **Rejoining Decision Formula**:
   ```
   keep_weights = (missed_rounds <= max_missed_rounds)
   ```
   Where `max_missed_rounds` is a configurable threshold (default: 3 rounds)

4. **Client Tracking**: Each client has a unique persistent ID that's maintained across failures and reconnections.

## Running the System

### Prerequisites
- Python 3.7+
- PyTorch
- Flower (flwr)
- torchvision
- matplotlib

### Basic Usage

```bash
python launcher.py
```

This will start the server and 3 clients with default parameters.

## Customizing Parameters

The launcher script supports various command-line arguments:

```bash
python launcher.py --num_clients 5 --rounds 20 --dataset_sizes 5000,8000,10000,15000,20000 --fail_probs 0.2,0.1,0.3,0.15,0.25 --recovery_probs 0.7,0.8,0.6,0.75,0.9
```

### Available Parameters:

- `--num_clients`: Number of clients to start (default: 3)
- `--rounds`: Number of federated learning rounds (default: 10)
- `--dataset_sizes`: Comma-separated list of dataset sizes for each client (default: "10000,20000,30000")
- `--fail_probs`: Comma-separated list of failure probabilities for each client (default: "0.2,0.1,0.3")
- `--recovery_probs`: Comma-separated list of recovery probabilities for each client (default: "0.7,0.8,0.6")

### Client-Specific Parameters

Each client can be configured individually:

- **Dataset Size**: Number of MNIST samples the client will use
- **Failure Probability**: Chance of client failure in each round (0.0-1.0)
- **Recovery Probability**: Chance of client recovery after failure (0.0-1.0)

Example for a specific client:

```bash
python client.py --client_id client_1 --dataset_size 15000 --fail_prob 0.15 --recovery_prob 0.8 --server 127.0.0.1:8080
```

## Visualization

The server automatically generates four types of visualizations:

1. **Global Loss**: Loss of the global model over rounds
2. **Global Accuracy**: Accuracy of the global model over rounds
3. **Weight Evolution**: Shows how model weights change during training
4. **Client Contributions**: Tracks each client's contribution over time, with clear markers for failed rounds

Visualizations are saved in the `results/` directory. 