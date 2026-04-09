import torch
import torch.nn as nn
import torch.nn.functional as F

def functional_model(
    x,
):
    return x / torch.mean(torch.abs(x), dim=1, keepdim=True)
batch_size = 32768
dim = 65535

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

