import torch
import torch.nn as nn
import torch.nn.functional as F

def functional_model(
    x,
    *,
    conv1d_weight,
    conv1d_bias,
    conv1d_stride,
    conv1d_padding,
    conv1d_dilation,
    conv1d_groups,
):
    return F.conv1d(x, conv1d_weight, conv1d_bias, stride=conv1d_stride, padding=conv1d_padding, dilation=conv1d_dilation, groups=conv1d_groups)
batch_size = 32
in_channels = 64
out_channels = 128
kernel_size = 3
length = 131072

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization

def get_inputs():
    x = torch.rand(batch_size, in_channels, length)
    return [x]

