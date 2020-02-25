#!/usr/local/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.layers.Flatten import Flatten
from nets.layers.Output import Output

import math

e = 2.71828

# basic short circuit model
# parameterization will happen later, if this even works
class ShortCircuitNet(nn.Module):
    def __init__(self, confidence_level, num_classes = 10, kernel_size = 3, dropout = 0.2, write_file=None):
        super(ShortCircuitNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size)
        self.conv2 = nn.Conv2d(32, 64, kernel_size)
        self.conv3 = nn.Conv2d(64, 128, kernel_size)

        self.fc1 = Output([7200, 256, 10])
        self.fc2 = Output([12544, 256, 10])
        self.fc3 = Output([21632, 256, 10])

        self.flatten = Flatten()
        self.dropout1 = nn.Dropout2d(dropout)

        self.log_confidence_level = math.log(confidence_level)

        self.should_short_circuit = True # this is just to easily toggle between the two

        self.write_file = write_file

    def forward(self, x):
        # print('----')
        # print(self.log_confidence_level)
        x = self.conv1(x)
        x = F.relu(x)
        if self.should_short_circuit:
            y = self.fc1(x)
            if max(y[0]).item() >= self.log_confidence_level:
                self.write_file.write('L1\n')
                self.write_file.write(str(max(y[0]).item()) + '\n')
                return y
        
        x = self.conv2(x)
        x = F.relu(x)
        if self.should_short_circuit:
            y = self.fc2(x)
            if max(y[0]) >= self.log_confidence_level:
                self.write_file.write('L2\n')
                self.write_file.write(str(max(y[0]).item()) + '\n')
                return y

        x = self.conv3(x)
        x = F.relu(x)
        y = self.fc3(x)
        if self.should_short_circuit:
            self.write_file.write('L3\n')
            self.write_file.write(str(max(y[0]).item()) + '\n')
        return y

