"""Copy Task Parameters."""
from attr import attrs, attrib, Factory
from torch import nn
from torch import optim
import torch
import math


@attrs
class CopyTaskParams(object):
    name = attrib(default="copy-task")
    controller_size = attrib(default=100, convert=int)
    controller_layers = attrib(default=1,convert=int)
    num_heads = attrib(default=1, convert=int)
    sequence_width = attrib(default=8, convert=int)
    sequence_min_len = attrib(default=1,convert=int)
    sequence_max_len = attrib(default=20, convert=int)
    memory_n = attrib(default=128, convert=int)
    memory_m = attrib(default=20, convert=int)
    num_batches = attrib(default=50000, convert=int)
    batch_size = attrib(default=100, convert=int)
    rmsprop_lr = attrib(default=1e-4, convert=float)
    rmsprop_momentum = attrib(default=0.9, convert=float)
    rmsprop_alpha = attrib(default=0.95, convert=float)
    loss = attrib(default=nn.BCELoss())
    save_every = attrib(default=100, convert=int)
    illustrate_every = attrib(default=100, convert=int)

    def get_illustrative_sample(self):
        '''Sequence will represent a rectified sine wave'''
        seq_len = self.sequence_max_len
        seq = torch.zeros(seq_len, 1, self.sequence_width)
        for i in range(seq_len):
            seq[i,0,int(self.sequence_width*abs(math.sin(2*i*math.pi/seq_len)))%self.sequence_width] = 1.0
        inp = torch.zeros(seq_len + 1, 1, self.sequence_width + 1)
        inp[:seq_len, :, :self.sequence_width] = seq
        inp[seq_len, :, self.sequence_width] = 1.0  # delimiter in our control channel
        outp = seq.clone()

        return inp, outp
