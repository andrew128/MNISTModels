#!/usr/local/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.layers.Output import Output

# 1 conv layer
class Simple(nn.Module):
    def __init__(self):
        super(Simple, self).__init__()

        input_channels = 1
        self.conv1 = nn.Conv2d(input_channels, 32, 3)

    # this model architecture follows v0 all_digit_model.py
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        return x