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
import logging
import random
from matplotlib.lines import Line2D
import time

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
    "client_dataset_size": {},
    "expected_clients": set(),
    "first_round_clients": set()
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
    def __init__(self, *args, min_available_clients=1, min_fit_clients=1, **kwargs):
        super().__init__(*args, min_available_clients=min_available_clients, 
                         min_fit_clients=min_fit_clients, **kwargs)
        self.client_history = {}
        self.current_round = 0
        self.verbose = True
        self.clients_initialized = False
        self.initial_wait_time = 5  # Time to wait for initial connections
    
    def initialize_clients(self, client_manager):
        """Initialize all clients to ensure they start at round 1"""
        # Wait for clients to connect
        logger.info(f"Waiting {self.initial_wait_time}s for initial client connections...")
        time.sleep(self.initial_wait_time)
        
        # Register all available clients
        available_clients = client_manager.all()
        global_metrics["expected_clients"] = set(available_clients)
        global_metrics["first_round_clients"] = set(available_clients)
        
        logger.info(f"Initializing {len(available_clients)} clients for round 1")
        
        for client_id in available_clients:
            # Setup client status tracking
            global_metrics["client_status"][client_id] = {
                "active": True, "missed_rounds": 0, "last_active_round": 0
            }
            
            # Initialize reliability with round 1 entry
            global_metrics["client_reliability"][client_id] = [{
                "round": 1, "reliability": 0.5, 
                "status": "pending", "data_recovered": False
            }]
            
            # Setup history tracking
            self.client_history[client_id] = {"participation_records": []}
        
        self.clients_initialized = True
        return available_clients
    
    def configure_fit(self, server_round, parameters, client_manager):
        """Configure clients for training with rejoining support"""
        self.current_round = server_round
        
        # Initialize clients if first round
        if not self.clients_initialized:
            self.initialize_clients(client_manager)
        
        # Get available clients
        config = {}
        available_clients = client_manager.all()
        logger.info(f"Round {server_round}: {len(available_clients)} clients available")
        
        # Add any new clients that weren't seen before
        self._handle_new_clients(available_clients, server_round)
        
        # Check if any clients rejoined after failure
        for client_id in available_clients:
            if client_id in global_metrics["client_status"]:
                status = global_metrics["client_status"][client_id]
                
                # Initialize status if needed
                if "active" not in status:
                    status["active"] = True
                if "missed_rounds" not in status:
                    status["missed_rounds"] = 0
                
                # Client is rejoining after failure
                if status.get("active") == False:
                    self._handle_rejoining_client(client_id, server_round, config)
        
        # Add round to config
        fit_config = {"round": server_round}
        
        # Update fit_config with client-specific configs
        if config:
            for client_id, client_config in config.items():
                fit_config[f"client_{client_id}"] = client_config
        
        # Let the parent method handle client selection
        return super().configure_fit(server_round, parameters, client_manager)
    
    def _handle_new_clients(self, available_clients, server_round):
        """Initialize new clients that weren't seen before"""
        for client_id in available_clients:
            if client_id not in global_metrics["client_status"]:
                # New client
                global_metrics["client_status"][client_id] = {
                    "active": True, "missed_rounds": 0, "last_active_round": 0
                }
                
                # Ensure round 1 entry exists
                if client_id not in global_metrics["client_reliability"]:
                    global_metrics["client_reliability"][client_id] = [{
                        "round": 1, "reliability": 0.5, 
                        "status": "pending", "data_recovered": False
                    }]
                
                self.client_history[client_id] = {"participation_records": []}
                global_metrics["expected_clients"].add(client_id)
                
                # Add to first round clients if this is round 1
                if server_round == 1:
                    global_metrics["first_round_clients"].add(client_id)
    
    def _handle_rejoining_client(self, client_id, server_round, config):
        """Handle a client rejoining after failure"""
        status = global_metrics["client_status"][client_id]
        
        # Calculate reliability and determine weight keeping
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
            "status": "rejoin",
            "keep_weights": keep_weights
        })
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate results with failure handling"""
        # Update round tracking
        global_metrics["rounds"] = server_round
        
        # Record client participation and statuses
        self._record_client_participation(server_round, results, failures)
        
        # Process results for global metrics if we have valid data
        self._process_global_metrics(server_round, results)
        
        # Handle clients not in results or failures
        self._handle_missing_clients(server_round, results, failures)
        
        # Fill in gaps for visualization
        self._fill_missing_rounds(server_round)
        
        # Return aggregated parameters
        if not results:
            return None, {}
            
        try:
            return super().aggregate_fit(server_round, results, failures)
        except ZeroDivisionError:
            logger.error(f"Division by zero in aggregation for round {server_round}")
            return None, {}
    
    def _record_client_participation(self, server_round, results, failures):
        """Record successful and failed client participation"""
        # Record successful clients
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            
            # Initialize history if needed
            if client_id not in self.client_history:
                self.client_history[client_id] = {"participation_records": []}
                
            # Calculate contribution safely
            total_examples = sum(r.num_examples for _, r in results)
            contribution = fit_res.num_examples / max(1, total_examples)
            
            # Record success
            self._record_client_success(client_id, server_round, contribution)
        
        # Record failed clients
        for client_proxy in failures:
            client_id = client_proxy.cid
            
            # Initialize history if needed
            if client_id not in self.client_history:
                self.client_history[client_id] = {"participation_records": []}
                
            # Record failure
            self._record_client_failure(client_id, server_round)
    
    def _record_client_success(self, client_id, server_round, contribution):
        """Record successful client participation"""
        # Add participation record
        self.client_history[client_id]["participation_records"].append({
            "round": server_round,
            "status": "success",
            "contribution": contribution,
        })
        
        # Update client status
        if client_id not in global_metrics["client_status"]:
            global_metrics["client_status"][client_id] = {
                "active": True, "missed_rounds": 0, 
                "last_active_round": server_round
            }
        else:
            global_metrics["client_status"][client_id]["active"] = True
            global_metrics["client_status"][client_id]["missed_rounds"] = 0
            global_metrics["client_status"][client_id]["last_active_round"] = server_round
        
        # Calculate reliability
        reliability_score, _ = self.calculate_client_reliability_score(client_id, server_round)
        
        # Update visualization data
        self._update_reliability_data(
            client_id, server_round, "success", 
            reliability_score, False
        )
    
    def _record_client_failure(self, client_id, server_round):
        """Record client failure"""
        # Add failure record
        self.client_history[client_id]["participation_records"].append({
            "round": server_round,
            "status": "failure"
        })
        
        # Update client status
        if client_id not in global_metrics["client_status"]:
            global_metrics["client_status"][client_id] = {
                "active": False, "missed_rounds": 1, 
                "last_active_round": 0
            }
        else:
            global_metrics["client_status"][client_id]["active"] = False
            global_metrics["client_status"][client_id]["missed_rounds"] += 1
        
        # Calculate reliability
        reliability_score, _ = self.calculate_client_reliability_score(client_id, server_round)
        
        # Determine if data can be recovered
        data_recovered = reliability_score > 0.7
        
        # Update visualization data
        self._update_reliability_data(
            client_id, server_round, "failure", 
            reliability_score, data_recovered
        )
    
    def _process_global_metrics(self, server_round, results):
        """Process aggregated results for global metrics"""
        if not results:
            return
            
        num_examples_total = sum(fit_res.num_examples for _, fit_res in results)
        if num_examples_total <= 0:
            return
            
        try:
            # Aggregate parameters
            parameters_aggregated, _ = super().aggregate_fit(server_round, results, [])
            if parameters_aggregated is None:
                return
                
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
        except ZeroDivisionError:
            logger.warning(f"⚠️ Division by zero in metrics calculation for round {server_round}")
    
    def _handle_missing_clients(self, server_round, results, failures):
        """Handle clients that are neither in results nor failures"""
        successful_clients = {client_proxy.cid for client_proxy, _ in results}
        failed_clients = {client_proxy.cid for client_proxy in failures}
        
        # Find clients that should participate but didn't
        expected_clients = global_metrics["expected_clients"].copy()
        missing_clients = expected_clients - successful_clients - failed_clients
        
        for client_id in missing_clients:
            if client_id in global_metrics["client_status"]:
                # Update status
                status = global_metrics["client_status"][client_id]
                status["active"] = False
                status["missed_rounds"] += 1
                
                # Add missed record
                if client_id not in self.client_history:
                    self.client_history[client_id] = {"participation_records": []}
                
                self.client_history[client_id]["participation_records"].append({
                    "round": server_round,
                    "status": "missed"
                })
                
                # Calculate reliability
                reliability_score, _ = self.calculate_client_reliability_score(client_id, server_round)
                
                # Update visualization data
                self._update_reliability_data(
                    client_id, server_round, "missed", 
                    reliability_score, False
                )
    
    def _update_reliability_data(self, client_id, round_num, status, reliability, data_recovered=False):
        """Update client reliability tracking data for visualization"""
        entries = [e for e in global_metrics["client_reliability"][client_id] 
                  if e["round"] == round_num]
        
        if entries and entries[0]["status"] == "pending":
            # Update existing entry
            entries[0]["status"] = status
            entries[0]["reliability"] = reliability
            if status == "failure":
                entries[0]["data_recovered"] = data_recovered
        else:
            # Create new entry
            global_metrics["client_reliability"][client_id].append({
                "round": round_num,
                "reliability": reliability,
                "status": status,
                "data_recovered": data_recovered if status == "failure" else False
            })
    
    def _fill_missing_rounds(self, current_round):
        """Fill in missing round entries for visualization continuity"""
        for client_id in global_metrics["expected_clients"]:
            # Ensure client has reliability data
            if client_id not in global_metrics["client_reliability"]:
                global_metrics["client_reliability"][client_id] = []
            
            reliability_data = global_metrics["client_reliability"][client_id]
            rounds_present = {entry["round"] for entry in reliability_data}
            
            # Ensure round 1 entry exists
            if 1 not in rounds_present:
                global_metrics["client_reliability"][client_id].append({
                    "round": 1,
                    "reliability": 0.5,
                    "status": "missed",
                    "data_recovered": False
                })
            
            # Fill other missing rounds
            for r in range(2, current_round + 1):
                if r not in rounds_present:
                    # Find previous round's reliability
                    prev_entries = [e for e in reliability_data if e["round"] < r]
                    prev_reliability = 0.5
                    
                    if prev_entries:
                        prev_entry = max(prev_entries, key=lambda e: e["round"])
                        prev_reliability = prev_entry["reliability"]
                    
                    # Add entry with slightly degraded reliability
                    global_metrics["client_reliability"][client_id].append({
                        "round": r,
                        "reliability": max(0.1, prev_reliability - 0.05),
                        "status": "missed",
                        "data_recovered": False
                    })
    
    def calculate_client_reliability_score(self, client_id, current_round):
        """Calculate client reliability score using exponential time decay"""
        # Get client history
        history = self.client_history.get(client_id, {"participation_records": []})
        participation_records = history.get("participation_records", [])
        
        if not participation_records:
            return 0.5, False  # Default for new clients
        
        # Constants
        decay_rate = 0.2
        max_history = 20
        
        # Calculate weighted score
        reliability_score = 0.0
        normalization_factor = 0.0
        
        for record in sorted(participation_records[-max_history:], key=lambda x: x["round"]):
            # Apply exponential decay based on time difference
            time_diff = current_round - record["round"]
            time_weight = math.exp(-decay_rate * time_diff)
            
            # Calculate impact based on record type
            if record["status"] == "success":
                contribution = record.get("contribution", 0.1)
                impact = 0.3 * (1.0 + min(1.0, contribution * 2.0))
            elif record["status"] == "failure":
                impact = -1.0
            else:  # missed
                impact = -0.5
            
            reliability_score += impact * time_weight
            normalization_factor += time_weight
        
        # Normalize to 0-1 range
        if normalization_factor > 0:
            reliability_score = 0.5 + (reliability_score / (2.5 * normalization_factor))
            reliability_score = max(0.1, min(0.95, reliability_score))
        else:
            reliability_score = 0.5
        
        # Debug output
        if self.verbose and random.random() < 0.1:
            logger.info(f"Client {client_id[:8]}: reliability={reliability_score:.3f}")
        
        # Determine weight keeping
        last_seen = max([r["round"] for r in participation_records])
        rounds_missed = current_round - last_seen
        threshold = 0.6 * math.exp(-0.05 * rounds_missed)
        keep_weights = reliability_score >= threshold
        
        return reliability_score, keep_weights

def plot_global_metrics():
    """Create plots for global metrics (loss, accuracy, weights)"""
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

def plot_client_reliability():
    """Create client reliability visualization"""
    plt.figure(figsize=(14, 8))
    
    # Define markers for different states
    success_marker = 'o'       # Circle: Normal success
    failure_marker = 'x'       # X: Failure
    keep_weights_marker = '^'  # Triangle up: Recovered with original weights
    new_weights_marker = 'v'   # Triangle down: Recovered with new weights
    missed_marker = '.'        # Dot: Missed round
    
    # Create consistent colors for clients
    client_ids = list(global_metrics["client_reliability"].keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(client_ids))))
    client_colors = {cid: colors[i % len(colors)] for i, cid in enumerate(client_ids)}
    
    # Determine max round
    max_rounds = global_metrics["rounds"]
    
    # Plot lines for each client
    for client_id, reliability_data in global_metrics["client_reliability"].items():
        if not reliability_data:
            continue
        
        # Sort data and create continuous plotting data
        reliability_data.sort(key=lambda x: x["round"])
        rounds_data = {entry["round"]: entry["reliability"] for entry in reliability_data}
        
        continuous_rounds = list(range(1, max_rounds + 1))
        continuous_reliability = []
        
        for r in continuous_rounds:
            if r in rounds_data:
                continuous_reliability.append(rounds_data[r])
            else:
                # Find previous value or use default
                prev_values = [rounds_data[prev_r] for prev_r in rounds_data if prev_r < r]
                reliability = 0.5 if not prev_values else prev_values[-1] * 0.95
                continuous_reliability.append(reliability)
        
        # Plot line connecting all points
        plt.plot(continuous_rounds, continuous_reliability, '-', 
                color=client_colors[client_id], label=f"Client {client_id[:8]}", 
                alpha=0.7, linewidth=1.5)
    
    # Add markers for each data point
    for client_id, reliability_data in global_metrics["client_reliability"].items():
        client_color = client_colors[client_id]
        
        for entry in reliability_data:
            # Skip pending entries
            if entry.get("status", "") == "pending":
                continue
                
            # Determine marker and styling based on status
            marker = success_marker
            size = 80
            edge_color = 'black'
            line_width = 1.5
            
            if entry.get("status") == "failure":
                marker = failure_marker
                size = 100
                edge_color = 'red'
                line_width = 2
            elif entry.get("status") == "missed":
                marker = missed_marker
                size = 60
                edge_color = 'gray'
                line_width = 1
            elif entry.get("status") == "rejoin":
                if entry.get("keep_weights", False):
                    marker = keep_weights_marker
                    edge_color = 'green'
                else:
                    marker = new_weights_marker
                    edge_color = 'blue'
                size = 120
                line_width = 2
            
            # Add point with styling
            plt.scatter(
                entry["round"], 
                entry["reliability"],
                marker=marker,
                s=size,
                color=client_color,
                edgecolors=edge_color,
                linewidth=line_width,
                zorder=5 if entry.get("status") != "success" else 3
            )
    
    # Add legend elements
    legend_elements = [
        Line2D([0], [0], marker=success_marker, color='w', markerfacecolor='gray', 
              markeredgecolor='black', markersize=10, label='Success'),
        Line2D([0], [0], marker=failure_marker, color='w', markerfacecolor='gray', 
              markeredgecolor='red', markersize=10, label='Failure'),
        Line2D([0], [0], marker=missed_marker, color='w', markerfacecolor='gray', 
              markeredgecolor='gray', markersize=10, label='Missed'),
        Line2D([0], [0], marker=keep_weights_marker, color='w', markerfacecolor='gray', 
              markeredgecolor='green', markersize=10, label='Recover w/ Weights'),
        Line2D([0], [0], marker=new_weights_marker, color='w', markerfacecolor='gray', 
              markeredgecolor='blue', markersize=10, label='Recover w/ New Weights')
    ]
    
    # Create two legends
    client_legend = plt.legend(loc='upper left', fontsize=10)
    plt.gca().add_artist(client_legend)
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.title("Client Reliability Scores Over Rounds", fontsize=14)
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Reliability Score", fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to show integer rounds
    plt.xticks(range(1, max_rounds + 1))
    
    plt.tight_layout()
    plt.savefig("results/client_reliability.png", bbox_inches='tight', dpi=300)
    plt.close()

def print_client_summary():
    """Print summary of client participation"""
    print("\nClient participation summary:")
    for client_id, data in global_metrics["client_reliability"].items():
        # Sort data by round, then by time added (assuming later entries in the list are more recent)
        data_by_round = {}
        
        # Process entries in reverse order (most recent first) to ensure we get the last status for each round
        for entry in reversed(sorted(data, key=lambda e: e.get("round", 0))):
            if entry.get("status") != "pending":
                round_num = entry.get("round", 0)
                if round_num not in data_by_round:
                    data_by_round[round_num] = entry.get("status", "unknown")
        
        rounds = sorted(data_by_round.keys())
        statuses = {r: data_by_round[r] for r in rounds}
        print(f"Client {client_id[:8]}: Rounds {rounds} with statuses: {statuses}")

def create_visualizations():
    """Create all visualizations"""
    print(f"Creating visualizations with {len(global_metrics['loss'])} loss points")
    
    # Print client summary
    print_client_summary()
    
    # Create plots
    plot_global_metrics()
    plot_client_reliability()
    
    print("✅ All visualizations saved to 'results' directory")

def main():
    # Define strategy
    strategy = FedAvgWithFailureHandling(
        min_fit_clients=1,
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