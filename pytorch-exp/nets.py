#!/usr/local/bin/python
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        input_channels = 1
        self.conv1 = nn.Conv2d(input_channels, 32, 3)

        self.fc1 = nn.Linear(5408, 128)
        self.fc2 = nn.Linear(128, 10)

        self.flatten = Flatten()
        self.dropout1 = nn.Dropout2d(0.2)

    # this model architecture follows v0 all_digit_model.py
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

