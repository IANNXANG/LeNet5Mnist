import matplotlib.pyplot as plt



def main(model_name):
    # 读取损失值并可视化
    losses = []
    with open(f'./log/losses{model_name}.txt', 'r') as f:
        for line in f:
            losses.append(float(line.strip()))

    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(f'loss_plot_{model_name}.png')

if __name__ == '__main__':
    main("LeNet5")
    main("LeNet5Sigmoid")
    main("LeNet5HalfKernels")
    main("LeNet5WithDropout10")
    main("LeNet5WithDropout20")