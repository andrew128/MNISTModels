#!/usr/local/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.layers.convs.ConvLayer import ConvLayer
from nets.layers.Output import Output

class Conv1Net(nn.Module):
    def __init__(self, in_channels, out_channels, fc_layer_neurons):
        super(Conv1Net, self).__init__()

        self.conv1 = ConvLayer(in_channels, out_channels)
        self.simple_output = Output(fc_layer_neurons)

    def forward(self, x):
        x = self.conv1(x)
        x = self.simple_output(x)
        return x
