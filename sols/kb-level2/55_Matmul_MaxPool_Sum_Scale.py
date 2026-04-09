import torch
import torch.nn as nn
import torch.nn.functional as F

def functional_model(
    x,
    *,
    matmul_weight,
    matmul_bias,
    max_pool_kernel_size,
    max_pool_stride,
    max_pool_padding,
    max_pool_dilation,
    max_pool_ceil_mode,
    max_pool_return_indices,
    scale_factor,
):
    x = F.linear(x, matmul_weight, matmul_bias)
    x = F.max_pool1d(x.unsqueeze(1), kernel_size=max_pool_kernel_size, stride=max_pool_stride, padding=max_pool_padding, dilation=max_pool_dilation, ceil_mode=max_pool_ceil_mode, return_indices=max_pool_return_indices).squeeze(1)
    x = torch.sum(x, dim=1)
    x = x * scale_factor
    return x
batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]

def get_inputs():
    return [torch.rand(batch_size, in_features)]

