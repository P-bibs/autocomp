# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_013845/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'scale_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'scale_factor']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, scales the output, and then applies a minimum operation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

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
    if 'scale_factor' in flat_state:
        state_kwargs['scale_factor'] = flat_state['scale_factor']
    else:
        state_kwargs['scale_factor'] = getattr(model, 'scale_factor')
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- CUDA Kernel for Fused Operations ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

template <typename scalar_t>
__global__ void fused_conv_scale_min_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const scalar_t scale_factor,
    const int out_height,
    const int out_width) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int spatial_elements = out_height * out_width;
    
    if (tid >= batch_size * spatial_elements) return;
    
    const int batch = tid / spatial_elements;
    const int spatial_idx = tid % spatial_elements;
    const int out_h = spatial_idx / out_width;
    const int out_w = spatial_idx % out_width;
    
    // Calculate min across all channels for this spatial location
    scalar_t min_val = 0.0;
    bool initialized = false;
    
    for (int out_c = 0; out_c < out_channels; ++out_c) {
        // Calculate input coordinates for this output position and channel
        const int in_h_start = out_h * stride - padding;
        const int in_w_start = out_w * stride - padding;
        
        scalar_t conv_result = 0.0;
        
        // Perform convolution for this output position and channel
        const int group_id = out_c / (out_channels / groups);
        const int weight_offset_base = out_c * (in_channels / groups) * kernel_size * kernel_size;
        
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int in_h = in_h_start + kh * dilation;
                const int in_w = in_w_start + kw * dilation;
                
                if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                    for (int ic = group_id * (in_channels / groups); 
                         ic < (group_id + 1) * (in_channels / groups); ++ic) {
                        const int input_idx = ((batch * in_channels + ic) * height + in_h) * width + in_w;
                        const int weight_idx = weight_offset_base + (ic - group_id * (in_channels / groups)) * kernel_size * kernel_size + kh * kernel_size + kw;
                        conv_result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Add bias
        conv_result += bias[out_c];
        
        // Apply scale
        conv_result *= scale_factor;
        
        // Update minimum
        if (!initialized) {
            min_val = conv_result;
            initialized = true;
        } else {
            min_val = fminf(min_val, conv_result);
        }
    }
    
    output[tid] = min_val;
}

void fused_conv_scale_min_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const float scale_factor) {
    
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);
    const auto out_channels = weight.size(0);
    const auto kernel_size = weight.size(2);
    
    const int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    const int threads_per_block = 256;
    const int elements = batch_size * out_height * out_width;
    const int blocks = (elements + threads_per_block - 1) / threads_per_block;
    
    const at::cuda::CUDAGuard device_guard(input.device());
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_conv_scale_min_kernel", ([&] {
        fused_conv_scale_min_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            height,
            width,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            static_cast<scalar_t>(scale_factor),
            out_height,
            out_width
        );
    }));
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the function in the .cu file
void fused_conv_scale_min_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const float scale_factor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_scale_min", &fused_conv_scale_min_forward, "Fused conv, scale, and min operation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_scale_min_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
    scale_factor,
):
    # Calculate output dimensions
    batch_size = x.size(0)
    in_channels = x.size(1)
    height = x.size(2)
    width = x.size(3)
    out_channels = conv_weight.size(0)
    kernel_size = conv_weight.size(2)
    
    out_height = (height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_width = (width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    # Allocate output tensor
    output = torch.empty(batch_size, 1, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_conv_scale_min(
        x, conv_weight, conv_bias, output,
        conv_stride, conv_padding, conv_dilation, conv_groups, scale_factor
    )
    
    return output

batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
