import torch
from sklearn.cross_validation import _BaseKFold
from torch import nn
from controller import LSTMController
from rom import ROM
from state import State
import numpy as np


def _split_cols(mat, lengths):
    """Split a 2D matrix to variable length columns."""
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results


class ModelCell(nn.Module):
    def __init__(self, params):
        super(ModelCell, self).__init__()

        # set params
        self.params = params

        # create controller
        self.controller = LSTMController(params.sequence_width + 1,
                                         params.controller_size,
                                         params.controller_layers)

        # create memory
        self.memory = ROM(params.memory_n, params.memory_m)

        # create state
        self.state = State(self.memory, self.controller)

        # create FC layer for addressing using controller output
        self.addressing_params_sizes = [self.memory.M, 1, 1, 3, 1]
        self.fc1 = nn.Sequential(
            nn.Linear(params.controller_size, sum(self.addressing_params_sizes)),
            nn.Sigmoid()
        )

        # create FC layer to make output from controller output and read value
        self.fc2 = nn.Sequential(
            nn.Linear(params.controller_size + self.memory.M, params.sequence_width + 1),
            nn.Sigmoid()
        )

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def forward(self, X):
        cout, self.state.controllerstate.state = self.controller(X, self.state.controllerstate.state)
        address_params = self.fc1(cout)
        k, beta, g, s, gamma = _split_cols(address_params, self.addressing_params_sizes)
        self.state.readstate.w = self.memory.address(k, beta, g, s, gamma, self.state.readstate.w)
        self.state.readstate.r = self.memory.read(self.state.readstate.w)
        self.memory.write(X)
        outp = self.fc2(torch.cat([cout, self.state.readstate.r], dim=1))

        return outp












