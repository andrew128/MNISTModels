#!/usr/local/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.layers.convs.ConvLayer import ConvLayer
from nets.layers.Output import Output

# ConvLayer w/ max pooling
# functions as intermediate layer
class ConvStackerNet(nn.Module):
    def __init__(self, prev_convs):
        super(ConvStackerNet, self).__init__()

        self.prev_convs = prev_convs
        self.my_prev_convs = []
        for i in range(len(prev_convs)):
            index = str(i)
            exec("self.prev_layer" + index + " = prev_convs[" + index + "]")
            exec("self.my_prev_convs.append(self.prev_layer" + index + ")")

        self.last_conv = ConvLayer(32, 64)
        self.simple_output = Output(9216)

    def forward(self, x):
        for i in range(len(self.my_prev_convs)):
            x = self.my_prev_convs[i](x)
            x = F.relu(x)
        x = self.last_conv(x)
        x = self.simple_output(x)
        return x
