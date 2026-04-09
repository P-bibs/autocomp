import torch
import torch.nn as nn
import torch.nn.functional as F

def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
    max_pool_kernel_size,
    max_pool_stride,
    max_pool_padding,
    max_pool_dilation,
    max_pool_ceil_mode,
    max_pool_return_indices,
    global_avg_pool_output_size,
    divisor,
    bias,
    sum_dim,
):
    x = F.conv3d(x, conv_weight, conv_bias, stride=conv_stride, padding=conv_padding, dilation=conv_dilation, groups=conv_groups)
    x = x / divisor
    x = F.max_pool3d(x, kernel_size=max_pool_kernel_size, stride=max_pool_stride, padding=max_pool_padding, dilation=max_pool_dilation, ceil_mode=max_pool_ceil_mode, return_indices=max_pool_return_indices)
    x = F.adaptive_avg_pool3d(x, global_avg_pool_output_size)
    x = x + bias
    x = torch.sum(x, dim=sum_dim)
    return x
batch_size   = 128  
in_channels  = 8            
out_channels = 16  
depth = 16; height = width = 64 
depth = 16; height = width = 64 
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

