#!/usr/local/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.layers.Flatten import Flatten

# full convolutional model. 2 convolutions, used for benchmarking.
class BaseNet(nn.Module):
    def __init__(self, input_channels, middle_channels, final_channels, fc_layer_big, fc_layer_small,
     num_classes = 10, kernel_size = 3, dropout = 0.2):
        super(BaseNet, self).__init__()

        input_channels = 1
        self.conv1 = nn.Conv2d(input_channels, middle_channels, kernel_size)
        self.conv2 = nn.Conv2d(middle_channels, final_channels, kernel_size, 1)

        self.fc1 = nn.Linear(fc_layer_big, fc_layer_small)
        self.fc2 = nn.Linear(fc_layer_small, num_classes)

        self.flatten = Flatten()
        self.dropout1 = nn.Dropout2d(dropout)

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

