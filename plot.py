import matplotlib.pyplot as plt
import os


def main(model_name):
    # 读取损失值并可视化
    losses = []

    with open(f'./log/losses{model_name}.txt', 'r') as f:
        for line in f:
            losses.append(float(line.strip()))

    plt.clf()

    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(f'./png/loss_plot_{model_name}.png')

if __name__ == '__main__':
    if not os.path.exists('./png'):
        os.makedirs('./png')
    main("LeNet5") #原始LeNet5
    main("LeNet5Sigmoid") #使用sigmoid激活函数代替ReLU
    main("LeNet5HalfKernels") #使用一半的卷积核
    main("LeNet5WithDropout10") #使用dropout10%
    main("LeNet5WithDropout20") #使用dropout20%