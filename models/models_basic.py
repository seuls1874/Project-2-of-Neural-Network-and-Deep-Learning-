# model_basic.py
import torch.nn as nn
import torch.nn.functional as F

class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 输出：32x32x32
        self.pool = nn.MaxPool2d(2, 2)               # 输出：32x16x16
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 输出：64x16x16 → pool: 64x8x8

        # 全连接层
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + MaxPool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + MaxPool
        x = x.view(-1, 64 * 8 * 8)            # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
