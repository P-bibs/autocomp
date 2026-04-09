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
    subtract_value_1,
    subtract_value_2,
):
    x = F.conv2d(x, conv_weight, conv_bias, stride=conv_stride, padding=conv_padding, dilation=conv_dilation, groups=conv_groups)
    x = x - subtract_value_1
    x = x - subtract_value_2
    x = torch.nn.functional.mish(x)
    return x
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

