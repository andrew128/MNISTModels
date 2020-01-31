#!/usr/local/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.layers.Flatten import Flatten

# full simple 1 convolutional model
class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        input_channels = 1
        self.conv1 = nn.Conv2d(input_channels, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        self.flatten = Flatten()
        self.dropout1 = nn.Dropout2d(0.2)

    # this model architecture follows v0 all_digit_model.py
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

