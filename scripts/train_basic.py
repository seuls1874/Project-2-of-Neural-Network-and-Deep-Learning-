import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import matplotlib.pyplot as plt

# 添加父目录到 Python 路径，方便导入自定义模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.models_basic import BasicCNN  # 你的基础CNN模型
from data.loaders import get_cifar_loader  # 你自定义的加载函数

# ✅ 全局参数
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    # ✅ 加载数据
    trainloader = get_cifar_loader(root='./data', batch_size=BATCH_SIZE, train=True, shuffle=True)
    testloader = get_cifar_loader(root='./data', batch_size=BATCH_SIZE, train=False, shuffle=False)

    # ✅ 初始化模型、损失函数、优化器、以及训练和测试列表
    model = BasicCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loss_list = []
    test_acc_list = []

    # ✅ 开始训练
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # ✅ 测试集评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
        train_loss_list.append(running_loss)
        test_acc_list.append(accuracy)

    # ✅ 保存模型
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/basic_model.pth")
    print("✅ 模型已保存到 checkpoints/basic_model.pth")
    # 绘制 Loss 曲线
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), train_loss_list, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epoch')
    plt.legend()
    plt.savefig('outputs/train_loss_curve.png')  # 自动保存图像
    plt.close()

    # 绘制 Accuracy 曲线
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), test_acc_list, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy vs Epoch')
    plt.legend()
    plt.savefig('outputs/test_acc_curve.png')
    plt.close()

    print("✅ 曲线图已保存到 outputs/ 文件夹")