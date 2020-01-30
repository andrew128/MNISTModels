#!/usr/local/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.layers.convs.ConvLayer import ConvLayer
from nets.layers.Output import Output

class Conv1Net(nn.Module):
    def __init__(self):
        super(Conv1Net, self).__init__()

        self.conv1 = ConvLayer(1, 32)
        self.simple_output = Output(5408)

    def forward(self, x):
        x = self.conv1(x)
        x = self.simple_output(x)
        return x
