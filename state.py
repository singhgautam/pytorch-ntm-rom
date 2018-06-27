import torch
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
from torch.nn import Parameter
from torch import nn
import numpy as np

class ReadState(nn.Module):
    def __init__(self, memory):
        super(ReadState, self).__init__()
        self.memory = memory

    def reset(self, batch_size):
        self.w = torch.zeros(batch_size, self.memory.N)
        self.w[:,0] = 1.0 # set reader attention at first spot in the memory
        self.r = self.memory.read(self.w)

class ControllerState(nn.Module):
    def __init__(self, controller):
        super(ControllerState, self).__init__()
        self.controller = controller

        # starting hidden state is a learned parameter
        self.lstm_h_bias = Parameter(torch.randn(self.controller.num_layers, 1, self.controller.num_outputs) * 0.05)
        self.lstm_c_bias = Parameter(torch.randn(self.controller.num_layers, 1, self.controller.num_outputs) * 0.05)

    def reset(self, batch_size):
        h = self.lstm_h_bias.clone().repeat(1, batch_size, 1)
        c = self.lstm_c_bias.clone().repeat(1, batch_size, 1)
        self.state = h, c

class State(nn.Module):
    def __init__(self, memory, controller):
        super(State, self).__init__()
        self.memory = memory
        self.controller = controller

    def reset(self, batch_size):
        # setup readstate
        self.readstate = ReadState(self.memory)
        self.readstate.reset(batch_size)

        # setup controller state
        self.controllerstate = ControllerState(self.controller)
        self.controllerstate.reset(batch_size)
