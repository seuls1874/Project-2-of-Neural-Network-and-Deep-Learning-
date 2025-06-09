# scripts/landscape_vgg_a.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.loaders import get_cifar_loader
from models.vgg import VGG_A
from models.vgg_bn import VGG_BatchNorm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
LEARNING_RATES = [1e-4, 5e-4, 1e-3, 2e-3]


def train_and_collect_losses(model_class, model_name):
    all_loss_lists = []
    for lr in LEARNING_RATES:
        print(f"Training {model_name} with learning rate: {lr}")
        trainloader = get_cifar_loader(batch_size=128, train=True)
        model = model_class().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        epoch_losses = []
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
            epoch_losses.append(running_loss)
        all_loss_lists.append(epoch_losses)

    # 计算 max_curve 和 min_curve
    max_curve = [max(losses[i] for losses in all_loss_lists) for i in range(EPOCHS)]
    min_curve = [min(losses[i] for losses in all_loss_lists) for i in range(EPOCHS)]

    # 可视化
    os.makedirs("outputs", exist_ok=True)
    plt.figure()
    plt.plot(max_curve, label='Max Loss')
    plt.plot(min_curve, label='Min Loss')
    plt.fill_between(range(EPOCHS), min_curve, max_curve, alpha=0.3, label='Gap')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Landscape - {model_name}')
    plt.legend()
    out_path = f'outputs/{model_name.lower()}_landscape_true.png'
    plt.savefig(out_path)
    print(f"\u2705 Saved: {out_path}")


if __name__ == '__main__':
    train_and_collect_losses(VGG_A, "VGG_A")
    train_and_collect_losses(VGG_BatchNorm, "VGG_BatchNorm")
