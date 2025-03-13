# launcher.py
import argparse
import subprocess
import time
import signal
import sys
import os
import platform

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Launch Federated Learning System")
parser.add_argument("--num-clients", type=int, default=3, help="Number of clients")
parser.add_argument("--rounds", type=int, default=10, help="Number of federated learning rounds")
parser.add_argument("--port", type=int, default=8080, help="Server port")
args = parser.parse_args()

# List to keep track of processes
processes = []

# Function to terminate processes on exit
def cleanup_and_exit(signal=None, frame=None):
    print("\n🛑 Terminating all processes...")
    for p in processes:
        try:
            if platform.system() == 'Windows':
                p.terminate()
            else:
                import os
                os.kill(p.pid, signal.SIGKILL)
            print(f"Process {p.pid} terminated")
        except:
            pass
    print("Cleanup complete")
    sys.exit(0)

# Register signal handlers for clean shutdown
signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)

# Create directories
os.makedirs("results", exist_ok=True)
os.makedirs("client_states", exist_ok=True)

# Check Flower version and downgrade if needed
try:
    import flwr
    flwr_version = flwr.__version__
    print(f"Current Flower version: {flwr_version}")
    
    # If newer than 1.4.0, downgrade
    if flwr_version > "1.4.0":
        print("⚠️ Newer Flower version detected. Downgrading to 1.4.0...")
        subprocess.run([sys.executable, "-m", "pip", "install", "flwr==1.4.0"], check=True)
        print("✅ Downgraded to Flower 1.4.0")
        print("Please restart this script for the changes to take effect.")
        sys.exit(0)
except Exception as e:
    print(f"Error checking/downgrading Flower version: {e}")
    print("Proceeding with current installation...")

# Client configurations
client_configs = [
    {"id": "client1", "dataset_size": 10000, "failure_prob": 0.1, "recovery_prob": 0.7},
    {"id": "client2", "dataset_size": 20000, "failure_prob": 0.2, "recovery_prob": 0.5}, 
    {"id": "client3", "dataset_size": 30000, "failure_prob": 0.05, "recovery_prob": 0.9}
]

# First clean up any existing state files to avoid KeyErrors
print("🧹 Cleaning up old state files...")
for config in client_configs:
    state_file = f"client_states/client_{config['id']}.json"
    if os.path.exists(state_file):
        os.remove(state_file)
        print(f"  Removed {state_file}")

# Start the server
print("🚀 Starting Federated Learning server...")
server_cmd = [
    sys.executable, "server.py",
    "--rounds", str(args.rounds),
    "--min-clients", "1",
    "--port", str(args.port)
]
server_proc = subprocess.Popen(server_cmd)
processes.append(server_proc)

# Wait for server to initialize
time.sleep(3)
print(f"Server started on port {args.port}")

# Start the clients
print(f"🚀 Starting {args.num_clients} clients...")
for i, config in enumerate(client_configs[:args.num_clients]):
    print(f"  • Client {config['id']}: Dataset size={config['dataset_size']}, "
          f"Failure prob={config['failure_prob']}, Recovery prob={config['recovery_prob']}")
    
    client_cmd = [
        sys.executable, "client.py",
        "--id", config["id"],
        "--dataset-size", str(config["dataset_size"]),
        "--failure-prob", str(config["failure_prob"]),
        "--recovery-prob", str(config["recovery_prob"]),
        "--server-address", f"127.0.0.1:{args.port}"
    ]
    
    client_proc = subprocess.Popen(client_cmd)
    processes.append(client_proc)
    
    # Small delay between client starts
    time.sleep(1)

print("\n✅ All processes started")
print("Press Ctrl+C to stop the system")

try:
    # Wait for server to complete
    server_proc.wait()
    print("✅ Server finished execution")
    
    # Server already handles visualization upon completion
    
    # Terminate remaining clients
    for proc in processes[1:]:
        if proc.poll() is None:  # If process is still running
            if platform.system() == 'Windows':
                proc.terminate()
            else:
                import os
                os.kill(proc.pid, signal.SIGKILL)
            
except KeyboardInterrupt:
    cleanup_and_exit()

print("🎉 Federated learning process completed!")