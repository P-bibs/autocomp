# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_132110/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'min_value', 'divisor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'min_value', 'divisor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

    def forward(self, *args) -> torch.Tensor:
        return functional_model(*args, **extract_state_kwargs(self))


def build_reference_model():
    init_inputs = list(get_init_inputs())
    model = ModelNew(*init_inputs)
    model.eval()
    return model


def extract_state_kwargs(model):
    flat_state = {}
    for name, value in model.named_parameters():
        flat_state[name.replace('.', '_')] = value
    for name, value in model.named_buffers():
        flat_state[name.replace('.', '_')] = value
    state_kwargs = {}
    init_inputs = list(get_init_inputs())
    init_arg_map = {name: value for name, value in zip(INIT_PARAM_NAMES, init_inputs)}
    # State for conv_transpose (nn.ConvTranspose3d)
    if 'conv_transpose_weight' in flat_state:
        state_kwargs['conv_transpose_weight'] = flat_state['conv_transpose_weight']
    else:
        state_kwargs['conv_transpose_weight'] = getattr(model.conv_transpose, 'weight', None)
    if 'conv_transpose_bias' in flat_state:
        state_kwargs['conv_transpose_bias'] = flat_state['conv_transpose_bias']
    else:
        state_kwargs['conv_transpose_bias'] = getattr(model.conv_transpose, 'bias', None)
    state_kwargs['conv_transpose_stride'] = model.conv_transpose.stride
    state_kwargs['conv_transpose_padding'] = model.conv_transpose.padding
    state_kwargs['conv_transpose_output_padding'] = model.conv_transpose.output_padding
    state_kwargs['conv_transpose_groups'] = model.conv_transpose.groups
    state_kwargs['conv_transpose_dilation'] = model.conv_transpose.dilation
    if 'min_value' in flat_state:
        state_kwargs['min_value'] = flat_state['min_value']
    else:
        state_kwargs['min_value'] = getattr(model, 'min_value')
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
    missing = [name for name in REQUIRED_STATE_NAMES if name not in state_kwargs]
    if missing:
        raise RuntimeError(f'Missing required state entries: {missing}')
    return state_kwargs


def get_functional_inputs():
    model = build_reference_model()
    forward_args = tuple(get_inputs())
    state_kwargs = extract_state_kwargs(model)
    return forward_args, state_kwargs




import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused conv_transpose3d + clamp + division
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_conv_transpose3d_clamp_div_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int groups,
    int dilation_d, int dilation_h, int dilation_w,
    float min_value,
    float divisor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = batch_size * out_channels * out_d * out_h * out_w;
    
    if (idx >= total_out) return;
    
    // Decompose output index
    int b = idx / (out_channels * out_d * out_h * out_w);
    int remainder = idx % (out_channels * out_d * out_h * out_w);
    int oc = remainder / (out_d * out_h * out_w);
    int remainder2 = remainder % (out_d * out_h * out_w);
    int od = remainder2 / (out_h * out_w);
    int remainder3 = remainder2 % (out_h * out_w);
    int oh = remainder3 / out_w;
    int ow = remainder3 % out_w;
    
    float val = 0.0f;
    
    // Add bias
    if (bias != nullptr) {
        val = bias[oc];
    }
    
    int group_id = oc / (out_channels / groups);
    int in_channels_per_group = in_channels / groups;
    
    // Convolution accumulation
    for (int ic = 0; ic < in_channels_per_group; ic++) {
        for (int kd = 0; kd < kernel_d; kd++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int id = od - output_padding_d + padding_d - kd * dilation_d;
                    int ih = oh - output_padding_h + padding_h - kh * dilation_h;
                    int iw = ow - output_padding_w + padding_w - kw * dilation_w;
                    
                    if (id % stride_d == 0 && ih % stride_h == 0 && iw % stride_w == 0) {
                        id /= stride_d;
                        ih /= stride_h;
                        iw /= stride_w;
                        
                        if (id >= 0 && id < in_d && ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            int input_idx = b * (in_channels * in_d * in_h * in_w) +
                                          (group_id * in_channels_per_group + ic) * (in_d * in_h * in_w) +
                                          id * (in_h * in_w) + ih * in_w + iw;
                            
                            int weight_idx = (group_id * (out_channels / groups) + oc % (out_channels / groups)) * 
                                           (in_channels_per_group * kernel_d * kernel_h * kernel_w) +
                                           ic * (kernel_d * kernel_h * kernel_w) +
                                           kd * (kernel_h * kernel_w) + kh * kernel_w + kw;
                            
                            val += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Fused clamp and division
    val = fmaxf(val, min_value);
    val = val / divisor;
    
    output[idx] = val;
}

void fused_conv_transpose3d_clamp_div_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int groups,
    int dilation_d, int dilation_h, int dilation_w,
    float min_value,
    float divisor
) {
    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    int total_out = batch_size * out_channels * out_d * out_h * out_w;
    int threads_per_block = 256;
    int num_blocks = (total_out + threads_per_block - 1) / threads_per_block;
    
    fused_conv_transpose3d_clamp_div_kernel<<<num_blocks, threads_per_block>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w, out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        groups, dilation_d, dilation_h, dilation_w,
        min_value, divisor
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_clamp_div_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int groups,
    int dilation_d, int dilation_h, int dilation_w,
    float min_value,
    float divisor
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward", &fused_conv_transpose3d_clamp_div_forward, 
          "Fused conv_transpose3d + clamp + division kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose3d',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False
)

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
    min_value,
    divisor,
):
    batch_size = x.shape[0]
    in_channels = x.shape[1]
    in_d, in_h, in_w = x.shape[2], x.shape[3], x.shape[4]
    
    out_channels = conv_transpose_weight.shape[1]
    kernel_d, kernel_h, kernel_w = conv_transpose_weight.shape[2], conv_transpose_weight.shape[3], conv_transpose_weight.shape[4]
    
    # Handle parameter normalization
    def normalize_param(param, ndim):
        if isinstance(param, int):
            return (param,) * ndim
        elif len(param) == 1:
            return (param[0],) * ndim
        elif len(param) == ndim:
            return tuple(param)
        else:
            raise ValueError(f"Invalid parameter length: {param}")
    
    stride_d, stride_h, stride_w = normalize_param(conv_transpose_stride, 3)
    padding_d, padding_h, padding_w = normalize_param(conv_transpose_padding, 3)
    output_padding_d, output_padding_h, output_padding_w = normalize_param(conv_transpose_output_padding, 3)
    dilation_d, dilation_h, dilation_w = normalize_param(conv_transpose_dilation, 3)
    
    # Calculate output dimensions
    out_d = (in_d - 1) * stride_d - 2 * padding_d + dilation_d * (kernel_d - 1) + output_padding_d + 1
    out_h = (in_h - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + output_padding_h + 1
    out_w = (in_w - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + output_padding_w + 1
    
    # Allocate output tensor
    output = torch.zeros(batch_size, out_channels, out_d, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Call fused kernel
    fused_ext.fused_forward(
        x, conv_transpose_weight, conv_transpose_bias, output,
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w, out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        conv_transpose_groups, dilation_d, dilation_h, dilation_w,
        min_value, divisor
    )
    
    return output

# Test parameters
batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 24, 48, 48
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
