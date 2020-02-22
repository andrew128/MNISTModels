#!/usr/local/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.layers.Flatten import Flatten

# Fully connected output layer
class Output(nn.Module):
    def __init__(self, fc_layer_neurons, dropout = 0.2):
        super(Output, self).__init__()

        self.flatten = Flatten()

        # setting up fully connected layers
        # fc_layer_neurons is an array, and should have at least starting and output neurons
        assert len(fc_layer_neurons) >= 2
        self.my_fc_layers = []
        for i in range(len(fc_layer_neurons) - 1):
            index = str(i)
            temp_fc = nn.Linear(fc_layer_neurons[i], fc_layer_neurons[i + 1])
            exec("self.fc" + index + " = temp_fc")
            exec("self.my_fc_layers.append(self.fc" + index + ")")
        
        self.dropout1 = nn.Dropout2d(dropout)
    
    def forward(self, x):
        x = F.max_pool2d(x, 2)
        x = self.flatten(x)
        x = self.my_fc_layers[0](x)

        for i in range(1, len(self.my_fc_layers)):
            x = F.relu(x)
            x = self.dropout1(x)
            x = self.my_fc_layers[i](x)

        x = F.log_softmax(x, dim=1)
        return x