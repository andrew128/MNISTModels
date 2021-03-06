#!/usr/local/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.layers.Flatten import Flatten

# toy model used for random experiments
class BaseNet(nn.Module):
    def __init__(self, input_channels, middle_channels, final_channels, fc_layer_big, fc_layer_small,
     num_classes = 10, kernel_size = 3, dropout = 0.2):
        super(BaseNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, middle_channels, kernel_size)
        self.conv2 = nn.Conv2d(middle_channels, final_channels, kernel_size)
        self.conv3 = nn.Conv2d(final_channels, 128, kernel_size)

        self.fc1 = nn.Linear(21632, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.flatten = Flatten()
        self.dropout1 = nn.Dropout2d(dropout)

    # this model architecture follows v0 all_digit_model.py
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout1(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

