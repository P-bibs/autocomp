import torch
import torch.nn as nn
import torch.nn.functional as F

def functional_model(
    x,
):
    return torch.tanh(x)
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []  # No special initialization inputs needed

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

