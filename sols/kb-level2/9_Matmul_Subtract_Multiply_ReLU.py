import torch
import torch.nn as nn
import torch.nn.functional as F

def functional_model(
    x,
    *,
    linear_weight,
    linear_bias,
    subtract_value,
    multiply_value,
):
    x = F.linear(x, linear_weight, linear_bias)
    x = x - subtract_value
    x = x * multiply_value
    x = torch.relu(x)
    return x
batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]

def get_inputs():
    return [torch.rand(batch_size, in_features)]

