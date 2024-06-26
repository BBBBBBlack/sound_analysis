# Cnn.py
import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # 输入为单层图像
                out_channels=16,  # 卷积成16层
                kernel_size=5,  # 卷积壳5x5
                stride=1,  # 步长，每次移动1步
                padding=2,  # 边缘层，给图像边缘增加像素值为0的框
            ),
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2),  # 池化层，将图像长宽减少一半
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


def save_model(net, path):
    torch.save(net, path)


def load_model(path):
    net = torch.load(path)
    return net
