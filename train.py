import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from lenet5_model import LeNet5, LeNet5Sigmoid, LeNet5HalfKernels, LeNet5WithDropout10, LeNet5WithDropout20, LeNet5For20x20
import torch.optim as optim
import torch.nn.functional as F
import os
import cv2


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

    if model_name == "LeNet5For20x20":
        transform = transforms.Compose([
            transforms.Resize((20, 20)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
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
    elif model_name == "LeNet5For20x20":
        model = LeNet5For20x20().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(1, 11):  # 训练10个周期
        train(model, device, train_loader, optimizer, epoch, model_name)

    # 保存模型
    torch.save(model.state_dict(), f'./model/model_{model_name}.pth')


if __name__ == '__main__':
    if not os.path.exists('./log'):
        os.makedirs('./log')

    if not os.path.exists('./model'):
        os.makedirs('./model')


    main("LeNet5") #原始LeNet5
    main("LeNet5Sigmoid") #使用sigmoid激活函数代替ReLU
    main("LeNet5HalfKernels") #使用一半的卷积核
    main("LeNet5WithDropout10") #使用dropout10%
    main("LeNet5WithDropout20") #使用dropout20%
    main("LeNet5For20x20") #使用20x20的图片

