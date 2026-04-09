import torch
import torch.nn as nn
import torch.nn.functional as F

def functional_model(
    x,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    max_pool1_kernel_size,
    max_pool1_stride,
    max_pool1_padding,
    max_pool1_dilation,
    max_pool1_ceil_mode,
    max_pool1_return_indices,
    max_pool2_kernel_size,
    max_pool2_stride,
    max_pool2_padding,
    max_pool2_dilation,
    max_pool2_ceil_mode,
    max_pool2_return_indices,
):
    x = F.conv_transpose3d(x, conv_transpose_weight, conv_transpose_bias, stride=conv_transpose_stride, padding=conv_transpose_padding, output_padding=conv_transpose_output_padding, groups=conv_transpose_groups, dilation=conv_transpose_dilation)
    x = F.max_pool3d(x, kernel_size=max_pool1_kernel_size, stride=max_pool1_stride, padding=max_pool1_padding, dilation=max_pool1_dilation, ceil_mode=max_pool1_ceil_mode, return_indices=max_pool1_return_indices)
    x = F.max_pool3d(x, kernel_size=max_pool2_kernel_size, stride=max_pool2_stride, padding=max_pool2_padding, dilation=max_pool2_dilation, ceil_mode=max_pool2_ceil_mode, return_indices=max_pool2_return_indices)
    x = torch.sum(x, dim=1, keepdim=True)
    return x
batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 32, 32, 32
kernel_size = 5
stride = 2
padding = 2

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

