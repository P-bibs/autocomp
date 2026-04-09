import torch
import torch.nn as nn
import torch.nn.functional as F

def functional_model(
    x,
    *,
    maxpool_kernel_size,
    maxpool_stride,
    maxpool_padding,
    maxpool_dilation,
    maxpool_ceil_mode,
    maxpool_return_indices,
):
    return F.max_pool2d(x, kernel_size=maxpool_kernel_size, stride=maxpool_stride, padding=maxpool_padding, dilation=maxpool_dilation, ceil_mode=maxpool_ceil_mode, return_indices=maxpool_return_indices)
batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]

def get_inputs():
    x = torch.rand(batch_size, channels, height, width)
    return [x]

