""" Custom RNN implementation """

import torch
from torch import nn, jit
import math


class RNNCell_base(jit.ScriptModule):     # (nn.Module):
#     __constants__ = ['bias']
    
    def __init__(self, input_size, hidden_size, nonlinearity, bias):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity

        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size)) 
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_ih, a=math.sqrt(5))    #, nonlinearity=nonlinearity)
        nn.init.kaiming_uniform_(self.weight_hh, a=math.sqrt(5))    #, nonlinearity=nonlinearity)
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_ih)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

            
class RNNCell(RNNCell_base):  # Euler integration of rate-neuron network dynamics 
    def __init__(self, input_size, hidden_size, nonlinearity = None, decay = 0.9, bias = True):
        super().__init__(input_size, hidden_size, nonlinearity, bias)
        self.decay = decay    #  torch.exp( - dt/tau )

    def forward(self, input, hidden):                        
        activity = self.nonlinearity(input @ self.weight_ih.t() +  hidden @ self.weight_hh.t() + self.bias)
        hidden   = self.decay * hidden + (1 - self.decay) * activity
        return hidden


class RNNLayer(nn.Module):    
    def __init__(self, *args):
        super().__init__()
        self.rnncell = RNNCell(*args)

    def forward(self, input, hidden_init):
        inputs = input.unbind(0)     # inputs has dimension [Time, batch, n_input]
        hidden = hidden_init[0]      # initial state has dimension [1, batch, n_input]
        outputs = []
        for i in range(len(inputs)):  # looping over the time dimension 
            hidden = self.rnncell(inputs[i], hidden)
            outputs += [hidden]       # vanilla RNN directly outputs the hidden state
        return torch.stack(outputs), hidden

