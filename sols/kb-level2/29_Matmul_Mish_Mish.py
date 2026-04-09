import torch
import torch.nn as nn
import torch.nn.functional as F

def functional_model(
    x,
    *,
    linear_weight,
    linear_bias,
):
    x = F.linear(x, linear_weight, linear_bias)
    x = torch.nn.functional.mish(x)
    x = torch.nn.functional.mish(x)
    return x
batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features)]

