import torch
import torch.nn as nn
import torch.nn.functional as F

def functional_model(
    A,
    B,
):
    return torch.matmul(A, B)
M = 16384 * 2
N = 16 * 2

def get_init_inputs():
    return []  # No special initialization inputs needed

def get_inputs():
    A = torch.rand(M, N)
    B = torch.rand(N, M)
    return [A, B]

