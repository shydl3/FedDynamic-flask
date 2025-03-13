import subprocess
import time
import argparse
import os
import signal
import sys

# Parse arguments
parser = argparse.ArgumentParser(description="Launch federated learning simulation")
parser.add_argument("--num_clients", type=int, default=3, help="Number of clients")
parser.add_argument("--rounds", type=int, default=10, help="Number of training rounds")
parser.add_argument("--dataset_sizes", type=str, default="10000,20000,30000", 
                   help="Comma-separated dataset sizes for each client")
parser.add_argument("--fail_probs", type=str, default="0.2,0.1,0.3",
                   help="Comma-separated failure probabilities")
parser.add_argument("--recovery_probs", type=str, default="0.7,0.8,0.6",
                   help="Comma-separated recovery probabilities")
parser.add_argument("--flwr_slient", action="store_true", help="Run Flower in silent mode")
args = parser.parse_args()

# Create results directory
os.makedirs("results", exist_ok=True)

if args.flwr_slient:
    os.environ["FLWR_LOG_LEVEL"] = "ERROR"
    os.environ["FLWR_LOG_LEVEL"] = "ERROR"
else:
    os.environ["FLWR_LOG_LEVEL"] = "INFO"
    os.environ["FLWR_LOG_LEVEL"] = "INFO"

# Parse client parameters
dataset_sizes = [int(x) for x in args.dataset_sizes.split(",")]
fail_probs = [float(x) for x in args.fail_probs.split(",")]
recovery_probs = [float(x) for x in args.recovery_probs.split(",")]

# Ensure we have enough parameters for all clients
while len(dataset_sizes) < args.num_clients:
    dataset_sizes.append(10000)
while len(fail_probs) < args.num_clients:
    fail_probs.append(0.1)
while len(recovery_probs) < args.num_clients:
    recovery_probs.append(0.8)

# Store all processes
processes = []

def cleanup():
    print("Cleaning up processes...")
    for p in processes:
        if p.poll() is None:  # If process is still running
            p.terminate()

# Handle Ctrl+C
def signal_handler(sig, frame):
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

try:
    # Start server
    print("🚀 Starting server...")
    server_cmd = ["python", "server.py"]
    server_process = subprocess.Popen(server_cmd)
    processes.append(server_process)
    
    # Wait for server to initialize
    time.sleep(2)
    
    # Start clients
    for i in range(args.num_clients):
        print(f"🚀 Starting client {i+1}...")
        client_cmd = [
            "python", "client.py",
            "--client_id", f"client_{i+1}",
            "--dataset_size", str(dataset_sizes[i]),
            "--fail_prob", str(fail_probs[i]),
            "--recovery_prob", str(recovery_probs[i])
        ]
        client_process = subprocess.Popen(client_cmd)
        processes.append(client_process)
        time.sleep(0.5)  # Small delay between starting clients
    
    print(f"✅ Started server and {args.num_clients} clients")
    print("⏳ Training in progress...")
    
    # Wait for server to complete
    server_process.wait()
    print("✅ Server has completed training")
    
    # Terminate any remaining client processes
    cleanup()

except Exception as e:
    print(f"Error: {e}")
    cleanup()