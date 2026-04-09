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
    bias,
):
    x = F.conv_transpose2d(x, conv_transpose_weight, conv_transpose_bias, stride=conv_transpose_stride, padding=conv_transpose_padding, output_padding=conv_transpose_output_padding, groups=conv_transpose_groups, dilation=conv_transpose_dilation)
    x = x - bias
    x = torch.tanh(x)
    return x
batch_size = 32
in_channels  = 64  
out_channels = 64  
height = width = 256 
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

