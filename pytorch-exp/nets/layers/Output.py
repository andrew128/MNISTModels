#!/usr/local/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.layers.Flatten import Flatten

class Output(nn.Module):
    def __init__(self, num_inputs, num_fc_layers = 2, dropout = 0.2):
        super(Output, self).__init__()

        self.num_inputs = num_inputs 
        self.num_fc_layers = num_fc_layers 

        self.flatten = Flatten()

        # TODO: loop number of fc layers
        # TODO: parameterize fc neurons
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout1 = nn.Dropout2d(dropout)
    
    def forward(self, x):
        x = F.max_pool2d(x, 2)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x