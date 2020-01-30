#!/usr/local/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.layers.convs.ConvLayer import ConvLayer
from nets.layers.Output import Output

# ConvLayer w/ max pooling
# functions as intermediate layer
class Conv2Net(nn.Module):
    def __init__(self, last_convs):
        super(Conv2Net, self).__init__()
        self.conv = ConvLayer(32, 64, prev_convs = last_convs)
        self.simple_output = Output(9216)

    def forward(self, x):
        x = self.conv(x)
        x = self.simple_output(x)
        return x
