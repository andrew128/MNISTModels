#!/usr/local/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.layers.Output import Output

# ConvLayer w/ max pooling
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, prev_convs = []):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3)
        self.prev_convs = prev_convs

    def forward(self, x):
        for c in self.prev_convs:
            x = c(x)
        x = self.conv(x)
        return x
        