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
from matplotlib.lines import Line2D
import logging
import random
import time
import threading
from flask import Flask, jsonify, render_template

start_time = time.time()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger("FedServer")

# Create results directory
os.makedirs("results", exist_ok=True)

# Global tracking variables
global_metrics = {
    "rounds": 0,
    "loss": [],
    "accuracy": [],
    "weights_evolution": [],
    "client_reliability": defaultdict(list),
    "client_status": {},
    "expected_clients": set()
}

def evaluate_global_model(parameters):
    """Evaluate the global model on the test dataset"""
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

class FedAvgWithFailureHandling(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_history = {}
        self.current_round = 0
        self.verbose = True
        self.clients_initialized = False
        self.initial_wait_time = 5  # Time to wait for initial connections
    
    def initialize_clients(self, client_manager):
        """Initialize all clients at the beginning"""
        logger.info(f"Waiting {self.initial_wait_time}s for initial client connections...")
        time.sleep(self.initial_wait_time)
        
        available_clients = client_manager.all()
        global_metrics["expected_clients"] = set(available_clients)
        
        logger.info(f"Initializing {len(available_clients)} clients for round 1")
        
        for client_id in available_clients:
            # Initialize client tracking
            global_metrics["client_status"][client_id] = {
                "active": True, "missed_rounds": 0, "last_active_round": 0
            }
            
            # Initialize reliability with round 1 entry
            global_metrics["client_reliability"][client_id] = [{
                "round": 1, "reliability": 0.5, 
                "status": "pending", "data_recovered": False
            }]
            
            self.client_history[client_id] = {"participation_records": []}
        
        self.clients_initialized = True
    
    def configure_fit(self, server_round, parameters, client_manager):
        """Configure clients for training with rejoining support"""
        self.current_round = server_round
        
        # Initialize clients if first round
        if not self.clients_initialized:
            self.initialize_clients(client_manager)
        
        # Get available clients and their configuration
        config = {}
        available_clients = client_manager.all()
        logger.info(f"Round {server_round}: {len(available_clients)} clients available")
        
        # Check if any clients rejoined after failure
        for client_id in available_clients:
            if client_id in global_metrics["client_status"]:
                status = global_metrics["client_status"][client_id]
                
                # Client is rejoining after failure - key fix: explicitly check if client was inactive
                if status.get("active") == False:
                    self._handle_rejoining_client(client_id, server_round, config)
                    logger.info(f"Detected client {client_id[:8]} rejoining in round {server_round}")
            else:
                # New client
                global_metrics["client_status"][client_id] = {
                    "active": True, "missed_rounds": 0, "last_active_round": 0
                }
                global_metrics["client_reliability"][client_id] = [{
                    "round": 1, "reliability": 0.5, "status": "pending"
                }]
                self.client_history[client_id] = {"participation_records": []}
                global_metrics["expected_clients"].add(client_id)
        
        # Add round to config
        fit_config = {"round": server_round}
        
        # Add client-specific configs
        for client_id, client_config in config.items():
            fit_config[f"client_{client_id}"] = client_config
        
        return super().configure_fit(server_round, parameters, client_manager)
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate results with failure handling"""
        global_metrics["rounds"] = server_round
        
        # Separate real successes from failures (client returns 0 examples when it fails)
        successful_results = []
        failed_clients = list(failures)
        
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            
            # If client reported 0 examples or "failed" status, it's a failure
            if fit_res.num_examples == 0 or (fit_res.metrics and fit_res.metrics.get("status") == "failed"):
                # Add to failures list
                if client_proxy not in failed_clients:
                    failed_clients.append(client_proxy)
                
                # Record failure
                self._record_client_failure(client_id, server_round)
                logger.info(f"Detected client {client_id[:8]} failure from metrics in round {server_round}")
            else:
                # Record genuine success
                successful_results.append((client_proxy, fit_res))
                self._record_client_success(client_id, server_round, fit_res.num_examples)
        
        # Log participation counts
        logger.info(f"Round {server_round}: {len(successful_results)} successful, {len(failed_clients)} failed")
        
        # Process global metrics if we have successful clients
        if successful_results:
            try:
                parameters_aggregated, _ = super().aggregate_fit(server_round, successful_results, [])
                if parameters_aggregated is not None:
                    ndarrays = fl.common.parameters_to_ndarrays(parameters_aggregated)
                    
                    # Evaluate global model
                    loss, accuracy = evaluate_global_model(ndarrays)
                    
                    # Store metrics
                    global_metrics["loss"].append(loss)
                    global_metrics["accuracy"].append(accuracy)
                    
                    # Store weight evolution
                    if ndarrays:
                        global_metrics["weights_evolution"].append(float(np.mean(ndarrays[0])))
                    
                    logger.info(f"📊 Round {server_round}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
            except Exception as e:
                logger.error(f"Error evaluating model in round {server_round}: {e}")
        
        # Handle missing clients (not in successes or failures)
        self._handle_missing_clients(server_round, successful_results, failed_clients)
        
        # Fill in missing rounds for visualization continuity
        self._fill_missing_rounds(server_round)
        
        # Return aggregated parameters for successful clients only
        if not successful_results:
            return None, {}
        
        try:
            return super().aggregate_fit(server_round, successful_results, [])
        except Exception as e:
            logger.error(f"Error in aggregation for round {server_round}: {e}")
            return None, {}
    
    def _record_client_success(self, client_id, server_round, num_examples):
        """Record successful client participation"""
        # Add to history
        self.client_history[client_id]["participation_records"].append({
            "round": server_round,
            "status": "success",
            "num_examples": num_examples
        })
        
        # Update status
        if client_id not in global_metrics["client_status"]:
            global_metrics["client_status"][client_id] = {
                "active": True, "missed_rounds": 0, "last_active_round": server_round
            }
        else:
            global_metrics["client_status"][client_id]["active"] = True
            global_metrics["client_status"][client_id]["missed_rounds"] = 0
            global_metrics["client_status"][client_id]["last_active_round"] = server_round
        
        # Calculate reliability and update visualization data
        reliability_score, _ = self.calculate_client_reliability_score(client_id, server_round)
        global_metrics["client_reliability"][client_id].append({
            "round": server_round,
            "reliability": reliability_score,
            "status": "success",
            "data_recovered": False
        })
    
    def _record_client_failure(self, client_id, server_round):
        """Record client failure"""
        # Add to history
        self.client_history[client_id]["participation_records"].append({
            "round": server_round,
            "status": "failure"
        })
        
        # Update status
        if client_id not in global_metrics["client_status"]:
            global_metrics["client_status"][client_id] = {
                "active": False, "missed_rounds": 1, "last_active_round": 0
            }
        else:
            global_metrics["client_status"][client_id]["active"] = False
            global_metrics["client_status"][client_id]["missed_rounds"] += 1
        
        # Calculate reliability and update visualization data
        reliability_score, _ = self.calculate_client_reliability_score(client_id, server_round)
        global_metrics["client_reliability"][client_id].append({
            "round": server_round,
            "reliability": reliability_score,
            "status": "failure",
            "data_recovered": reliability_score > 0.7
        })
    
    def _handle_missing_clients(self, server_round, successful_results, failed_clients):
        """Handle clients that are neither in results nor failures"""
        successful_ids = {client_proxy.cid for client_proxy, _ in successful_results}
        failed_ids = {client_proxy.cid for client_proxy in failed_clients}
        
        # Find clients that should participate but didn't
        expected_clients = global_metrics["expected_clients"].copy()
        missing_clients = expected_clients - successful_ids - failed_ids
        
        for client_id in missing_clients:
            if client_id in global_metrics["client_status"]:
                # Record as failure
                self._record_client_failure(client_id, server_round)
                logger.info(f"Client {client_id[:8]} missing in round {server_round} (marked as failure)")
    
    def _handle_rejoining_client(self, client_id, server_round, config):
        """Handle a client rejoining after failure"""
        status = global_metrics["client_status"][client_id]
        
        # Calculate reliability score
        reliability_score, keep_weights = self.calculate_client_reliability_score(
            client_id, server_round
        )
        
        # Configure client based on reliability
        if keep_weights:
            logger.info(f"🔄 Client {client_id[:8]} rejoined with reliability {reliability_score:.2f} - keeping weights")
            config[client_id] = {"keep_weights": True}
        else:
            logger.info(f"🆕 Client {client_id[:8]} rejoined with reliability {reliability_score:.2f} - new weights")
            config[client_id] = {"keep_weights": False}
        
        # Mark as active again
        status["active"] = True
        
        # Record rejoin event for visualization
        global_metrics["client_reliability"][client_id].append({
            "round": server_round,
            "reliability": reliability_score,
            "status": "rejoin",  # Ensure this status is correctly set
            "keep_weights": keep_weights
        })
        
        # Log rejoin event for debugging
        logger.info(f"REJOIN RECORDED: Client {client_id[:8]} in round {server_round}, keep_weights={keep_weights}")

    def _fill_missing_rounds(self, current_round):
        """Fill in gaps for visualization continuity"""
        for client_id in global_metrics["expected_clients"]:
            reliability_data = global_metrics["client_reliability"].get(client_id, [])
            rounds_with_data = {entry["round"] for entry in reliability_data}

            # Fill in any missing rounds from 1 to current
            for r in range(1, current_round + 1):
                if r not in rounds_with_data:
                    # Find most recent entry before this round
                    prev_entries = [e for e in reliability_data if e["round"] < r]

                    if prev_entries:
                        # Get most recent entry
                        latest = max(prev_entries, key=lambda e: e["round"])
                        prev_reliability = latest.get("reliability", 0.5)

                        # Keep status consistent during failure periods
                        if latest.get("status") == "failure":
                            status = "failure"
                        elif latest.get("status") == "rejoin":
                            status = "success"  # After rejoin, assume success unless failure occurs
                        else:
                            status = "missed"
                    else:
                        prev_reliability = 0.5
                        status = "missed"

                    # Add entry with slightly degraded reliability
                    global_metrics["client_reliability"][client_id].append({
                        "round": r,
                        "reliability": max(0.1, prev_reliability - 0.05),
                        "status": status,
                        "keep_weights": False,
                        "data_recovered": False
                    })
    
    def calculate_client_reliability_score(self, client_id, current_round):
        """Calculate client reliability score using exponential time decay with improved factors"""
        # Get client history
        history = self.client_history.get(client_id, {"participation_records": []})
        participation_records = history.get("participation_records", [])
        
        if not participation_records:
            return 0.5, False  # Default moderate reliability for new clients
        
        # Constants
        decay_rate = 0.15      # Slightly reduced decay rate for smoother transitions
        max_history = 20       # Consider at most 20 recent records
        base_success_impact = 0.5    # Base positive impact for successful participation 
        failure_impact = -1.0        # Strong negative impact for failures
        missed_impact = -0.5         # Moderate negative impact for missed rounds
        dataset_factor = 0.3         # Weight factor for dataset size
        reliability_threshold = 0.6  # Constant threshold for weight keeping
        
        # Calculate weighted score
        reliability_score = 0.0
        normalization_factor = 0.0
        consecutive_failures = 0
        
        # Sort records by round to process them in chronological order
        sorted_records = sorted(participation_records[-max_history:], key=lambda x: x["round"])
        
        for record in sorted_records:
            # Apply exponential time decay - more recent records count more
            time_diff = current_round - record["round"]
            time_weight = math.exp(-decay_rate * time_diff)
            
            # Calculate impact based on record type and properties
            if record["status"] == "success":
                # For successful participation:
                # 1. Base positive impact
                # 2. Bonus based on dataset size (num_examples)
                dataset_size = record.get("num_examples", 100)  # Default if not specified
                dataset_bonus = min(1.0, dataset_size / 5000) * dataset_factor
                impact = base_success_impact + dataset_bonus
                consecutive_failures = 0  # Reset consecutive failures
            elif record["status"] == "failure":
                # For failures:
                # 1. Negative impact
                # 2. Additional penalty for consecutive failures
                consecutive_failures += 1
                consecutive_penalty = min(0.5, consecutive_failures * 0.1)  # Up to 0.5 extra penalty
                impact = failure_impact - consecutive_penalty
            else:  # missed or unknown
                impact = missed_impact
            
            # Add weighted impact to score
            reliability_score += impact * time_weight
            normalization_factor += time_weight
        
        # Normalize to 0-1 range
        if normalization_factor > 0:
            # Center around 0.5, scale based on weighted impacts
            reliability_score = 0.5 + (reliability_score / (3.0 * normalization_factor))
            # Clamp to reasonable range, avoiding extreme values
            reliability_score = max(0.05, min(0.95, reliability_score))
        else:
            # Default value if no records to process
            reliability_score = 0.5
        
        # Check if client has been absent too long
        last_seen = max([r["round"] for r in participation_records]) if participation_records else 0
        rounds_missed = current_round - last_seen
        
        # Log useful information
        if self.verbose and random.random() < 0.2:  # Only log occasionally to reduce noise
            logger.info(f"Client {client_id[:8]}: reliability={reliability_score:.3f}, missed={rounds_missed} rounds")
        
        # Use constant threshold as requested, but reduce reliability if client missed too many rounds
        if rounds_missed > 3:
            # Apply penalty for extended absence
            reliability_score = max(0.05, reliability_score - (rounds_missed - 3) * 0.05)
        
        # Determine whether to keep weights based on constant threshold
        keep_weights = reliability_score >= reliability_threshold
        
        return reliability_score, keep_weights

def print_client_summary():
    """Print summary of client participation"""
    print("\nClient participation summary:")
    print(global_metrics["client_reliability"])

def plot_client_reliability():
    """Create client reliability visualization"""
    plt.figure(figsize=(14, 8))
    
    # Define markers for different states
    success_marker = 'o'       # Circle: Normal success
    failure_marker = 'x'       # X: Failure
    keep_weights_marker = '^'  # Triangle up: Recovered with original weights
    new_weights_marker = 'v'   # Triangle down: Recovered with new weights
    
    # Create consistent colors for clients
    client_ids = list(global_metrics["client_reliability"].keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(client_ids))))
    client_colors = {cid: colors[i % len(colors)] for i, cid in enumerate(client_ids)}
    
    max_rounds = global_metrics["rounds"]
    
    # For each client, get entries for each round, prioritizing 'rejoin' status over other statuses
    client_round_data = {}
    for client_id, reliability_data in global_metrics["client_reliability"].items():
        round_to_entry = {}
        
        # First pass: Add all entries
        for entry in reliability_data:
            round_num = entry["round"]
            status = entry.get("status", "")
            
            # If we don't have an entry for this round yet, add it
            if round_num not in round_to_entry:
                round_to_entry[round_num] = entry
            # If this is a rejoin entry, it takes precedence
            elif status == "rejoin":
                round_to_entry[round_num] = entry
                print(f"Found REJOIN entry: Client {client_id[:8]}, Round {round_num}, Keep weights: {entry.get('keep_weights', False)}")
            # Otherwise, only update if the existing entry is not a rejoin entry
            elif round_to_entry[round_num].get("status") != "rejoin":
                round_to_entry[round_num] = entry
        
        client_round_data[client_id] = round_to_entry
    
    # First pass: Plot lines for each client
    for client_id, round_data in client_round_data.items():
        rounds = sorted(round_data.keys())
        reliability = [round_data[r]["reliability"] for r in rounds]
        
        plt.plot(rounds, reliability, '-', color=client_colors[client_id], 
                label=f"Client {client_id[:8]}", alpha=0.7, linewidth=1.5)
    
    # Second pass: Add markers for each status
    for client_id, round_data in client_round_data.items():
        client_color = client_colors[client_id]
        
        for round_num, entry in round_data.items():
            status = entry.get("status", "unknown")
            if status == "pending":
                continue
            
            # Determine marker and styling
            marker = success_marker
            size = 80
            line_width = 1.5
            
            if status == "failure":
                marker = failure_marker
                size = 100
            elif status == "rejoin":
                # Explicitly handle rejoin status
                if entry.get("keep_weights", False):
                    marker = keep_weights_marker
                else:
                    marker = new_weights_marker
                size = 120
                
                print(f"Plotting REJOIN marker at ({round_num}, {entry['reliability']})")
            
            # Add marker - using client_color for both face and edge for consistency
            plt.scatter(
                round_num,
                entry["reliability"],
                marker=marker,
                s=size,
                color=client_color,  # Single color for both marker and edge
                linewidth=line_width,
                zorder=5 if status != "success" else 3
            )
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker=success_marker, color='w', markerfacecolor='gray', 
              markeredgecolor='black', markersize=10, label='Success'),
        Line2D([0], [0], marker=failure_marker, color='w', markerfacecolor='gray', 
              markeredgecolor='red', markersize=10, label='Failure'),
        Line2D([0], [0], marker=keep_weights_marker, color='w', markerfacecolor='gray', 
              markeredgecolor='green', markersize=10, label='Recover w/ Weights'),
        Line2D([0], [0], marker=new_weights_marker, color='w', markerfacecolor='gray', 
              markeredgecolor='blue', markersize=10, label='Recover w/ New Weights')
    ]
    
    # Add legends - one for clients, one for markers
    client_legend = plt.legend(loc='upper left', fontsize=10)
    plt.gca().add_artist(client_legend)
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.title("Client Reliability Scores Over Rounds", fontsize=14)
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Reliability Score", fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, max_rounds + 1))
    
    plt.tight_layout()
    plt.savefig("results/client_reliability.png", bbox_inches='tight', dpi=300)
    plt.close()

def plot_global_metrics():
    """Create plots for global metrics"""
    metrics = ["loss", "accuracy", "weights_evolution"]
    titles = ["Global Model Loss", "Global Model Accuracy", "Weight Evolution"]
    
    for metric, title in zip(metrics, titles):
        plt.figure(figsize=(10, 6))
        if global_metrics[metric]:
            plt.plot(range(1, len(global_metrics[metric])+1), global_metrics[metric], marker='o')
        else:
            plt.text(0.5, 0.5, f"No {metric} data available", horizontalalignment='center')
        plt.title(f"{title} over Rounds")
        plt.xlabel("Round")
        plt.ylabel(title.split()[-1])
        plt.grid(True)
        plt.savefig(f"results/{metric}.png")
        plt.close()

def create_visualizations():
    """Create visualizations after training"""
    print(f"Creating visualizations with {len(global_metrics['loss'])} loss points")
    
    print_client_summary()
    plot_global_metrics()
    plot_client_reliability()
    
    print("✅ All visualizations saved to 'results' directory")


def start_dashboard_server():
    """
    runs a Flask server，retrieve realtime global_metrics (after each round)。
    """
    app = Flask(__name__)
    app.config["GLOBAL_METRICS"] = global_metrics
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # 禁用静态文件缓存

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/metrics")
    def get_metrics():
        # Calculate server uptime
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        # Get global_metrics
        gm = app.config["GLOBAL_METRICS"]

        gm_copy = dict(gm)
        if isinstance(gm_copy.get("expected_clients"), set):
            gm_copy["expected_clients"] = list(gm_copy["expected_clients"])

        # Include all relevant fields in client_reliability
        reliability_data = {
            client_id: [
                {
                    "round": entry["round"],
                    "reliability": entry["reliability"],
                    "status": entry.get("status", "success"),  # Ensure status is included
                    "keep_weights": entry.get("keep_weights", False)  # Ensure keep_weights is included
                }
                for entry in reliability
            ]
            for client_id, reliability in gm_copy.get("client_reliability", {}).items()
        }
        gm_copy["client_reliability"] = reliability_data

        gm_copy["server_uptime"] = f"{hours}h {minutes}m {seconds}s"

        return jsonify(gm_copy)


    def run_flask():
        app.run(host="127.0.0.1", port=80, debug=False)

    # Flask thread runs at background
    dashboard_thread = threading.Thread(target=run_flask, daemon=True)
    dashboard_thread.start()
    print("✅ Flask realtime dashboard is up，visit http://127.0.0.1:80/ to view the training process")


def main():
    strategy = FedAvgWithFailureHandling(
        min_fit_clients=1,
        min_available_clients=1,
        min_evaluate_clients=0
    )

    start_dashboard_server()
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )
    
    create_visualizations()

if __name__ == "__main__":
    main()