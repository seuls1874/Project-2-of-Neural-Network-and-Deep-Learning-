import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.loaders import get_cifar_loader
from models.vgg_bn import VGG_BatchNorm
from utils.loss_tracker import LossTracker

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 20
LR = 0.001

if __name__ == '__main__':
    print("Using device:", DEVICE)

    trainloader = get_cifar_loader(batch_size=BATCH_SIZE, train=True)
    testloader = get_cifar_loader(batch_size=BATCH_SIZE, train=False)

    model = VGG_BatchNorm().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loss_list = []
    test_acc_list = []
    tracker = LossTracker()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        tracker.start_new_epoch()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            tracker.update(loss.item())
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
    torch.save(model.state_dict(), "checkpoints/vgg_bn.pth")

    os.makedirs("outputs", exist_ok=True)
    plt.plot(range(1, EPOCHS + 1), train_loss_list, label='Loss')
    plt.plot(range(1, EPOCHS + 1), test_acc_list, label='Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('VGG_BatchNorm Training')
    plt.savefig("outputs/vgg_bn_curve.png")
    print("✅ 模型与图像已保存")

    # 保存 loss 曲线
    max_curve = tracker.get_max_curve()
    min_curve = tracker.get_min_curve()
    gap_curve = tracker.get_gap_curve()

    plt.figure()
    plt.plot(max_curve, label='Max Loss')
    plt.plot(min_curve, label='Min Loss')
    plt.plot(gap_curve, label='Gap (Max - Min)')
    plt.xlabel('Epoch')

    
    plt.title('VGG_BN Loss Landscape')
    plt.legend()
    plt.savefig('outputs/vgg_bn_landscape.png')