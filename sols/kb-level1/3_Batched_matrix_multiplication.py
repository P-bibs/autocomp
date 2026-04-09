import torch
import torch.nn as nn
import torch.nn.functional as F

def functional_model(
    A,
    B,
):
    return torch.bmm(A, B)
batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_init_inputs():
    return []  # No special initialization inputs needed

def get_inputs():
    A = torch.rand(batch_size, m, k)
    B = torch.rand(batch_size, k, n)
    return [A, B]

