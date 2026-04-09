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
    return F.max_pool1d(x, kernel_size=maxpool_kernel_size, stride=maxpool_stride, padding=maxpool_padding, dilation=maxpool_dilation, ceil_mode=maxpool_ceil_mode, return_indices=maxpool_return_indices)
batch_size = 64
features = 192
sequence_length = 65536
kernel_size = 8
stride      = 1
padding     = 4
dilation    = 3            
return_indices = False

def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]

def get_inputs():
    x = torch.rand(batch_size, features, sequence_length)
    return [x]

