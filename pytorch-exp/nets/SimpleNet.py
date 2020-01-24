#!/usr/local/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.layers.Simple import Simple
from nets.layers.Output import Output

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.simple_conv = Simple()
        self.simple_output = Output(5408)

    def forward(self, x):
        x = self.simple_conv(x)
        x = self.simple_output(x)
        return x
    
    def get_conv_layers():
        return self.simple_conv
    
    def get_output_layers():
        return self.simple_output


