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
    x = torch.min(x, dim=1, keepdim=True)[0]
    x = torch.tanh(x)
    x = torch.tanh(x)
    return x
batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

