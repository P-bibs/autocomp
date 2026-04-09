import torch
import torch.nn as nn
import torch.nn.functional as F

def functional_model(
    x,
    *,
    gemm_weight,
    bias,
):
    x = F.linear(x, gemm_weight, None)
    x = x + bias
    x = torch.relu(x)
    return x
batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)

def get_init_inputs():
    return [in_features, out_features, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_features)]

