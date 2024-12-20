import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from lenet5_model import LeNet5, LeNet5Sigmoid, LeNet5HalfKernels, LeNet5WithDropout10, LeNet5WithDropout20, LeNet5For20x20
import torch.optim as optim
import torch.nn.functional as F
from train import model_select


def test(model, device, test_loader, model_name):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(f'{model_name} Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)')

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

    dataset_test = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)

    model = model_select(model_name, device)

    model.load_state_dict(torch.load(f'./model/model_{model_name}.pth'))  # 加载训练好的模型参数

    test(model, device, test_loader, model_name)

if __name__ == '__main__':
    main("LeNet5") #原始LeNet5
    main("LeNet5Sigmoid") #使用sigmoid激活函数代替ReLU
    main("LeNet5Tanh")  # 使用tanh激活函数代替ReLU
    main("LeNet5HalfKernels") #使用一半的卷积核
    main("LeNet5WithDropout10") #使用dropout10%
    main("LeNet5WithDropout20") #使用dropout20%
    main("LeNet5For20x20") #使用20x20的图片