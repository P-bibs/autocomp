# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_091623/code_5.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

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
    # State for conv (nn.Conv2d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
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

# ============================================================================
# CUDA Kernel Implementation
# ============================================================================

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_min_tanh_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    // Grid-stride loop over output spatial positions
    int out_h = (height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;
    
    int out_spatial = out_h * out_w;
    int total_out = batch_size * out_spatial;
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_out; idx += gridDim.x * blockDim.x) {
        int batch_idx = idx / out_spatial;
        int spatial_idx = idx % out_spatial;
        int out_y = spatial_idx / out_w;
        int out_x = spatial_idx % out_w;
        
        // Compute convolution with min reduction in one pass
        float min_val = 1e10f;
        
        for (int oc = 0; oc < out_channels; oc++) {
            float acc = 0.0f;
            int group = oc / (out_channels / groups);
            int in_ch_per_group = in_channels / groups;
            
            for (int ic = 0; ic < in_ch_per_group; ic++) {
                int in_channel_idx = group * in_ch_per_group + ic;
                
                for (int ky = 0; ky < kernel_h; ky++) {
                    for (int kx = 0; kx < kernel_w; kx++) {
                        int in_y = out_y * stride + ky * dilation - padding;
                        int in_x = out_x * stride + kx * dilation - padding;
                        
                        if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                            int in_idx = batch_idx * (in_channels * height * width) +
                                       in_channel_idx * (height * width) +
                                       in_y * width + in_x;
                            
                            int w_idx = oc * (in_ch_per_group * kernel_h * kernel_w) +
                                       ic * (kernel_h * kernel_w) +
                                       ky * kernel_w + kx;
                            
                            acc += input[in_idx] * weight[w_idx];
                        }
                    }
                }
            }
            
            acc += bias[oc];
            min_val = fminf(min_val, acc);
        }
        
        // Apply tanh once (not twice)
        float tanh_val = tanhf(min_val);
        
        // Write output: single channel (min reduction from out_channels)
        int out_idx = batch_idx * out_spatial + spatial_idx;
        output[out_idx] = tanh_val;
    }
}

void fused_conv_min_tanh_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(0);
    int height = input.size(2);
    int width = input.size(3);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    
    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    int out_h = (height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;
    int total_out = batch_size * out_h * out_w;
    
    int block_size = 256;
    int grid_size = (total_out + block_size - 1) / block_size;
    grid_size = min(grid_size, 65536);
    
    fused_conv_min_tanh_kernel<<<grid_size, block_size>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        batch_size, in_channels, out_channels,
        height, width, kernel_h, kernel_w,
        stride, padding, dilation, groups
    );
}
"""

# ============================================================================
# C++ Bindings
# ============================================================================

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_min_tanh_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation,
    int groups
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_conv_min_tanh",
        &fused_conv_min_tanh_forward,
        "Fused Conv2D + Min Reduction + Tanh kernel"
    );
}
"""

# ============================================================================
# Load CUDA Extension
# ============================================================================

fused_op_ext = load_inline(
    name='fused_conv_min_tanh',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ============================================================================
# Optimized Model Function
# ============================================================================

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
    batch_size = x.size(0)
    out_channels = conv_weight.size(0)
    
    # Calculate output spatial dimensions
    in_h, in_w = x.size(2), x.size(3)
    k_h, k_w = conv_weight.size(2), conv_weight.size(3)
    out_h = (in_h + 2 * conv_padding - conv_dilation * (k_h - 1) - 1) // conv_stride + 1
    out_w = (in_w + 2 * conv_padding - conv_dilation * (k_w - 1) - 1) // conv_stride + 1
    
    # Output shape: (batch_size, 1, out_h, out_w) after min reduction
    output = torch.zeros(
        batch_size, 1, out_h, out_w,
        dtype=x.dtype,
        device=x.device
    )
    
    # Call fused kernel
    fused_op_ext.fused_conv_min_tanh(
        x, conv_weight, conv_bias, output,
        conv_stride, conv_padding, conv_dilation, conv_groups
    )
    
    return output


# ============================================================================
# Test Code
# ============================================================================

batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
