import torch
import torch.nn as nn
import torch.nn.functional as F

def functional_model(
    A,
    B,
):
    return A.unsqueeze(1) * B
M = 4096
N = 4096

def get_init_inputs():
    return []  # No special initialization inputs needed

def get_inputs():
    A = torch.rand(N)
    B = torch.rand(N, M)
    return [A, B]

