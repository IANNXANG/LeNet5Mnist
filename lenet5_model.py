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

class LeNet5Tanh(nn.Module):
    def __init__(self):
        super(LeNet5Tanh, self).__init__()
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
        x = F.tanh(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 池化层
        x = F.tanh(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 池化层
        x = x.view(-1, 16 * 4 * 4)  # 展平
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
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
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class LeNet5For20x20(nn.Module):
    def __init__(self):
        super(LeNet5For20x20, self).__init__()
        # 第一层卷积层，输入1通道图像，输出6个特征图，核大小5x5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # 第二层卷积层，输入6个特征图，输出16个特征图，核大小5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # 计算经过两个卷积层和两个池化层后的尺寸
        # 输入尺寸: 20x20 -> conv1 (5x5) -> 16x16 -> max_pool2d (2x2) -> 8x8
        #           8x8 -> conv2 (5x5) -> 4x4 -> max_pool2d (2x2) -> 2x2
        # 因此，全连接层的输入应该是 16 * 2 * 2

        # 全连接层，输入16*2*2个神经元，输出120个神经元
        self.fc1 = nn.Linear(16 * 2 * 2, 120)
        # 全连接层，输入120个神经元，输出84个神经元
        self.fc2 = nn.Linear(120, 84)
        # 输出层，输入84个神经元，输出10个类别概率
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 输入图像尺寸: [batch_size, 1, 20, 20]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 池化层
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 池化层
        x = x.view(-1, 16 * 2 * 2)  # 展平，注意这里的尺寸要与conv2的输出相匹配
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


# 使用模型
model = LeNet5For20x20()
