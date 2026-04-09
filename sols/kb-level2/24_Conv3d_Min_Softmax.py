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
    dim,
):
    x = F.conv3d(x, conv_weight, conv_bias, stride=conv_stride, padding=conv_padding, dilation=conv_dilation, groups=conv_groups)
    x = torch.min(x, dim=dim)[0]
    x = torch.softmax(x, dim=1)
    return x
batch_size = 128
in_channels = 3
out_channels = 24  # Increased output channels
D, H, W = 24, 32, 32  # Increased depth
kernel_size = 3
dim = 2  # Dimension along which to apply minimum operation (e.g., depth)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

