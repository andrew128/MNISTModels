#!/usr/local/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.layers.Output import Output

# ConvLayer w/ max pooling
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super(ConvLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        return x