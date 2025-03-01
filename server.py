import flwr as fl
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import Net

# 记录全局模型损失和准确率
global_loss = []
global_acc = []
weights_over_time = []
client_contributions = {}

# 评估全局模型
def evaluate_global_model(parameters):
    """在本地 MNIST 测试集上评估全局模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)

    # 加载服务器端聚合后的模型参数
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

    # 加载 MNIST 测试数据集
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += loss_fn(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    avg_loss = loss / len(test_loader)
    
    global_loss.append(avg_loss)
    global_acc.append(acc)
    
    return avg_loss, acc

# 自定义 FedAvg 以进行可视化
class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化跟踪客户端首次加入轮次的字典
        self.client_first_round = {}
        # 保持原有的客户端贡献跟踪
        # 保持原有的权重变化跟踪
        self.weights_over_time = []
        self.rnd = 0
        
    def aggregate_fit(self, rnd, results, failures):
        """联邦训练轮次聚合"""
        if not results:
            return super().aggregate_fit(rnd, results, failures)

        new_clients = []
        for res in results:
            client_id = res[0].cid
            if client_id not in self.client_first_round:
                self.client_first_round[client_id] = rnd
                new_clients.append(client_id)

        if new_clients:
            print(f"🔔 Round {rnd} - New clients joined: {new_clients}")
            print(f"🔔 Total clients so far: {len(self.client_first_round)}")

        weighted_results = []
        total_weight = 0.0
        for res in results:
            client_id = res[0].cid
            # 计算客户端参与的轮次数
            participation_rounds = rnd - self.client_first_round[client_id] + 1
            # 使用对数函数使权重增长更加平滑
            weight = np.log2(participation_rounds + 1)
            total_weight += weight
            
            weighted_results.append((res[0], res[1], weight))
        
        if total_weight > 0:
            weighted_results = [
                (res[0], res[1], res[2] / total_weight) for res in weighted_results
            ]
        
        # 使用自定义权重聚合参数
        aggregated_parameters = self.aggregate_parameters_weighted(
            [
                (fl.common.parameters_to_ndarrays(res[1].parameters), res[2])
                for res in weighted_results
            ]
        )

        if aggregated_parameters:
            # 评估全局模型
            loss, acc = evaluate_global_model(aggregated_parameters)
            print(f"Round {rnd} - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
            
            # 记录权重变化
            mean_weight = np.mean(aggregated_parameters[0])
            self.weights_over_time.append(mean_weight)
            
            # 记录客户端贡献
            for res in results:
                client_id = res[0].cid
                client_parameters = fl.common.parameters_to_ndarrays(res[1].parameters)
                norm = np.linalg.norm(np.concatenate([p.flatten() for p in client_parameters]))
                if client_id not in client_contributions:
                    client_contributions[client_id] = []
                if len(client_contributions[client_id]) != self.rnd:
                    missing_rounds = rnd - len(client_contributions[client_id]) -1
                    if missing_rounds > 0: 
                        # 使用0或平均值填充缺失的轮次
                        fill_value = 0.0  # 或者使用平均值: np.mean(self.client_contributions[client_id])
                        client_contributions[client_id].extend([fill_value] * missing_rounds)
                client_contributions[client_id].append(norm)
                
        self.rnd += 1

        return fl.common.Parameters(
            tensors=fl.common.ndarrays_to_parameters(aggregated_parameters).tensors,
            tensor_type=fl.common.ndarrays_to_parameters(aggregated_parameters).tensor_type
        ), {}

    def aggregate_parameters_weighted(self, parameters_and_weights):
        """按权重聚合参数"""
        # 提取权重和参数
        parameters = [p[0] for p in parameters_and_weights]
        weights = [p[1] for p in parameters_and_weights]
        
        # 确保权重和为1
        weights = np.array(weights) / np.sum(weights) if np.sum(weights) > 0 else np.array(weights)
        
        # 初始化存储聚合参数的数组
        aggregated_parameters = [
            np.zeros_like(param) for param in parameters[0]
        ]
        
        # 按权重聚合参数
        for param_set, weight in zip(parameters, weights):
            for i, param in enumerate(param_set):
                aggregated_parameters[i] += param * weight
        
        return aggregated_parameters

# 启动服务器
if __name__ == "__main__":
    strategy = CustomFedAvg(
        fraction_fit=1.0,  # 100% 客户端参与
        min_fit_clients=2,  # **至少 1 个客户端**
        min_available_clients=2,  # **至少 3 个客户端连接**
    )

    print("🚀 服务器启动，等待 3 个客户端连接...")
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",  # 监听所有 IP
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )

    # ========================== #
    # 训练完成后可视化并保存图片  #
    # ========================== #

    # 📌 1. 绘制全局模型的损失曲线
    plt.figure(figsize=(6, 4))
    plt.plot(global_loss, label="Loss", color="blue")
    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.title("Global Model Loss over Rounds")
    plt.legend()
    plt.savefig("results/loss.png")
    plt.close()
    
    # 📌 2. 绘制全局模型的准确率曲线
    plt.figure(figsize=(6, 4))
    plt.plot(global_acc, label="Accuracy", color="green")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.title("Global Model Accuracy over Rounds")
    plt.legend()
    plt.savefig("results/accuracy.png")
    plt.close()

    # 📌 3. 修复 weights_over_time 记录后，绘制参数变化曲线
    plt.figure(figsize=(6, 4))
    plt.plot(weights_over_time, label="Mean Weights", color="red")
    plt.xlabel("Rounds")
    plt.ylabel("Mean Weight Value")
    plt.title("Weight Evolution in Federated Learning")
    plt.legend()
    plt.savefig("results/weights.png")
    plt.close()

    # 📌 4. 绘制不同客户端的贡献曲线
    plt.figure(figsize=(6, 4))
    for client_id, updates in client_contributions.items():
        plt.plot(updates, label=f"Client {client_id[:4]}")
    plt.xlabel("Rounds")
    plt.ylabel("Update Norm")
    plt.title("Client Contribution Over Rounds")
    plt.legend()
    plt.savefig("results/client_contribution.png")
    plt.close()

    print("✅ 训练完成，所有可视化结果已保存！")
