import torch
import torch.nn as nn
import torch.nn.functional as F

def functional_model(
    x,
    *,
    avg_pool_kernel_size,
    avg_pool_stride,
    avg_pool_padding,
    avg_pool_ceil_mode,
    avg_pool_count_include_pad,
):
    return F.avg_pool1d(x, kernel_size=avg_pool_kernel_size, stride=avg_pool_stride, padding=avg_pool_padding, ceil_mode=avg_pool_ceil_mode, count_include_pad=avg_pool_count_include_pad)
batch_size = 64
in_channels = 128
input_length = 65536
kernel_size = 8
stride = 1
padding = 4

def get_init_inputs():
    return [kernel_size, stride, padding]

def get_inputs():
    x = torch.rand(batch_size, in_channels, input_length)
    return [x]

