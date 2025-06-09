import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.loaders import get_cifar_loader
from models.models_bn_dropout import CNN_BN_Dropout

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    trainloader = get_cifar_loader(root='./data', batch_size=BATCH_SIZE, train=True)
    testloader = get_cifar_loader(root='./data', batch_size=BATCH_SIZE, train=False)

    model = CNN_BN_Dropout().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loss_list = []
    test_acc_list = []

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

        acc = 100 * correct / total
        train_loss_list.append(running_loss)
        test_acc_list.append(acc)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss:.2f}, Accuracy: {acc:.2f}%")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/cnn_bn_dropout.pth")

    # 可视化
    os.makedirs("outputs", exist_ok=True)
    plt.plot(range(1, EPOCHS + 1), train_loss_list, label='Loss')
    plt.plot(range(1, EPOCHS + 1), test_acc_list, label='Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('CNN_BN_Dropout Training')
    plt.savefig("outputs/cnn_bn_dropout.png")
    print("✅ 模型与图像已保存")
