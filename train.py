import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from lenet5_model import LeNet5, LeNet5Sigmoid, LeNet5HalfKernels, LeNet5WithDropout10, LeNet5WithDropout20
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def train(model, device, train_loader, optimizer, epoch, model_name):
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if batch_idx % 100 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    # 将损失值保存到文件
    with open(f'./log/losses{model_name}.txt', 'a') as f:
        for loss in losses:
            f.write(f"{loss}\n")


def main(model_name):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)

    if model_name == "LeNet5":
        model = LeNet5().to(device)
    elif model_name == "LeNet5Sigmoid":
        model = LeNet5Sigmoid().to(device)
    elif model_name == "LeNet5HalfKernels":
        model = LeNet5HalfKernels().to(device)
    elif model_name == "LeNet5WithDropout10":
        model = LeNet5WithDropout10().to(device)
    elif model_name == "LeNet5WithDropout20":
        model = LeNet5WithDropout20().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(1, 11):  # 训练10个周期
        train(model, device, train_loader, optimizer, epoch, model_name)

    # 保存模型
    torch.save(model.state_dict(), f'./model/model_{model_name}.pth')


if __name__ == '__main__':
    main("LeNet5")
    main("LeNet5Sigmoid")
    main("LeNet5HalfKernels")
    main("LeNet5WithDropout10")
    main("LeNet5WithDropout20")

