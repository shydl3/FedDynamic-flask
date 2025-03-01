import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import flwr as fl
from model import Net

# 设置服务器公网 IP（这里替换为你的 EC2 IP）
SERVER_IP = "128.195.54.86"  # 这里替换成你的 EC2 服务器公网 IP
SERVER_PORT = "8080"

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集（拆分 MNIST）
def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    
    # 按 50% 的比例拆分（两个客户端）
    trainset_part1, trainset_part2, trainset_part3 = torch.utils.data.random_split(trainset, [10000, 20000, 30000])

    if "client1" in __file__:
        return trainset_part1
    elif "client2" in __file__:
        return trainset_part2
    else:
        return trainset_part3

# 训练函数
def train(model, train_loader, epochs=5, lr=0.01):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

# 定义 FL 客户端
class FLClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = Net().to(device)
        self.train_loader = torch.utils.data.DataLoader(load_data(), batch_size=32, shuffle=True)

    def get_parameters(self, config=None):
        return [param.cpu().detach().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param).to(device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.train_loader, epochs=1)
        return self.get_parameters(), len(self.train_loader.dataset), {}

# 运行客户端（连接到服务器）
fl.client.start_numpy_client(server_address=f"{SERVER_IP}:{SERVER_PORT}", client=FLClient())

