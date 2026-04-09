import torch
import torch.nn as nn
import torch.nn.functional as F

def functional_model(
    x,
    *,
    dim,
):
    return torch.max(x, dim=dim)[0]
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_init_inputs():
    return [1] # Example, change to desired dimension

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2)
    return [x]

