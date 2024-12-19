import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 第一层卷积层，输入1通道图像，输出6个特征图，核大小5x5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # 第二层卷积层，输入6个特征图，输出16个特征图，核大小5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 全连接层，输入16*4*4个神经元，输出120个神经元
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # 全连接层，输入120个神经元，输出84个神经元
        self.fc2 = nn.Linear(120, 84)
        # 输出层，输入84个神经元，输出10个类别概率
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 输入图像尺寸: [batch_size, 1, 28, 28]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 池化层
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 池化层
        x = x.view(-1, 16 * 4 * 4)  # 展平
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class LeNet5Sigmoid(nn.Module):
    def __init__(self):
        super(LeNet5Sigmoid, self).__init__()
        # 第一层卷积层，输入1通道图像，输出6个特征图，核大小5x5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # 第二层卷积层，输入6个特征图，输出16个特征图，核大小5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 全连接层，输入16*4*4个神经元，输出120个神经元
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # 全连接层，输入120个神经元，输出84个神经元
        self.fc2 = nn.Linear(120, 84)
        # 输出层，输入84个神经元，输出10个类别概率
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 输入图像尺寸: [batch_size, 1, 28, 28]
        x = F.sigmoid(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 池化层
        x = F.sigmoid(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 池化层
        x = x.view(-1, 16 * 4 * 4)  # 展平
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class LeNet5HalfKernels(nn.Module):
    def __init__(self):
        super(LeNet5HalfKernels, self).__init__()
        # 第一层卷积层，输入1通道图像，输出3个特征图，核大小5x5
        self.conv1 = nn.Conv2d(1, 3, kernel_size=5)
        # 第二层卷积层，输入3个特征图，输出8个特征图，核大小5x5
        self.conv2 = nn.Conv2d(3, 8, kernel_size=5)
        # 全连接层，根据新的特征图尺寸调整输入神经元数
        self.fc1 = nn.Linear(8 * 4 * 4, 120)  # 假设输入是28x28，经过两次池化后为4x4
        # 全连接层，输入120个神经元，输出84个神经元
        self.fc2 = nn.Linear(120, 84)
        # 输出层，输入84个神经元，输出10个类别概率
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 输入图像尺寸: [batch_size, 1, 28, 28]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 池化层
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 池化层
        x = x.view(-1, 8 * 4 * 4)  # 展平，注意这里的尺寸要与conv2的输出相匹配
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class LeNet5WithDropout10(nn.Module):
    def __init__(self):
        super(LeNet5WithDropout10, self).__init__()
        # 第一层卷积层，输入1通道图像，输出6个特征图，核大小5x5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # 第二层卷积层，输入6个特征图，输出16个特征图，核大小5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # Dropout层，丢弃率10%
        self.dropout = nn.Dropout2d(p=0.1)

        # 全连接层，输入16*4*4个神经元，输出120个神经元
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # 全连接层，输入120个神经元，输出84个神经元
        self.fc2 = nn.Linear(120, 84)
        # 输出层，输入84个神经元，输出10个类别概率
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        # 输入图像尺寸: [batch_size, 1, 28, 28]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 池化层

        x = F.relu(self.conv2(x))
        x = self.dropout(x)  # 应用Dropout
        x = F.max_pool2d(x, 2)  # 池化层

        x = x.view(-1, 16 * 4 * 4)  # 展平

        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)  # 应用Dropout到全连接层
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class LeNet5WithDropout20(nn.Module):
    def __init__(self):
        super(LeNet5WithDropout20, self).__init__()
        # 第一层卷积层，输入1通道图像，输出6个特征图，核大小5x5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # 第二层卷积层，输入6个特征图，输出16个特征图，核大小5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # Dropout层，丢弃率20%
        self.dropout = nn.Dropout2d(p=0.2)

        # 全连接层，输入16*4*4个神经元，输出120个神经元
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # 全连接层，输入120个神经元，输出84个神经元
        self.fc2 = nn.Linear(120, 84)
        # 输出层，输入84个神经元，输出10个类别概率
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        # 输入图像尺寸: [batch_size, 1, 28, 28]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 池化层

        x = F.relu(self.conv2(x))
        x = self.dropout(x)  # 应用Dropout
        x = F.max_pool2d(x, 2)  # 池化层

        x = x.view(-1, 16 * 4 * 4)  # 展平

        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)  # 应用Dropout到全连接层
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
