import torch
import torch.nn as nn
import torch.nn.functional as F

def functional_model(
    predictions,
    targets,
):
    return torch.mean((predictions - targets) ** 2)
batch_size = 32768
input_shape = (32768,)

def get_init_inputs():
    return []

def get_inputs():
    scale = torch.rand(())
    return [torch.rand(batch_size, *input_shape)*scale, torch.rand(batch_size, *input_shape)]

