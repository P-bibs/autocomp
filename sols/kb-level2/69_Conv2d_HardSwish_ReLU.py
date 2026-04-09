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
):
    x = F.conv2d(x, conv_weight, conv_bias, stride=conv_stride, padding=conv_padding, dilation=conv_dilation, groups=conv_groups)
    x = torch.nn.functional.hardswish(x)
    x = torch.relu(x)
    return x
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

