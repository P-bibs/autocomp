# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_010751/code_2.py
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

# Custom CUDA kernel for fused convolution, scaling and min reduction
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

 template <typename scalar_t>
__global__ void fused_conv_scale_min_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_size,
    const int stride,
    const int padding,
    const scalar_t scale_factor,
    scalar_t* __restrict__ output) {
    
    // Output dimensions
    const int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    const int out_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    // Thread indices
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = batch_size * out_channels * out_h * out_w;
    
    if (tid >= total_threads) return;
    
    // Calculate indices
    const int out_w_idx = tid % out_w;
    const int out_h_idx = (tid / out_w) % out_h;
    const int out_c_idx = (tid / (out_w * out_h)) % out_channels;
    const int batch_idx = tid / (out_w * out_h * out_channels);
    
    // Input coordinates
    const int in_h_start = out_h_idx * stride - padding;
    const int in_w_start = out_w_idx * stride - padding;
    
    // Convolution computation
    scalar_t sum = 0;
    
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int in_h = in_h_start + kh;
                const int in_w = in_w_start + kw;
                
                // Check bounds
                if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                    const int input_idx = batch_idx * (in_channels * height * width) +
                                          ic * (height * width) +
                                          in_h * width + in_w;
                                          
                    const int weight_idx = out_c_idx * (in_channels * kernel_size * kernel_size) +
                                           ic * (kernel_size * kernel_size) +
                                           kh * kernel_size + kw;
                                           
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    sum += bias[out_c_idx];
    
    // Scale
    sum *= scale_factor;
    
    // Write to temporary output (before reduction)
    const int temp_out_idx = batch_idx * (out_channels * out_h * out_w) +
                             out_c_idx * (out_h * out_w) +
                             out_h_idx * out_w + out_w_idx;
    
    // For min reduction, we'll need to do a second pass
    // Store scaled conv result first
    output[temp_out_idx] = sum;
}

template <typename scalar_t>
__global__ void min_reduction_kernel(
    const scalar_t* __restrict__ input,
    const int batch_size,
    const int out_channels,
    const int out_h,
    const int out_w,
    scalar_t* __restrict__ output) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_h * out_w;
    
    if (tid >= total_elements) return;
    
    const int out_w_idx = tid % out_w;
    const int out_h_idx = (tid / out_w) % out_h;
    const int batch_idx = tid / (out_w * out_h);
    
    // Find minimum across channels
    scalar_t min_val = input[batch_idx * (out_channels * out_h * out_w) +
                             0 * (out_h * out_w) +
                             out_h_idx * out_w + out_w_idx];
                             
    for (int c = 1; c < out_channels; ++c) {
        scalar_t val = input[batch_idx * (out_channels * out_h * out_w) +
                             c * (out_h * out_w) +
                             out_h_idx * out_w + out_w_idx];
        if (val < min_val) min_val = val;
    }
    
    output[batch_idx * (out_h * out_w) +
           out_h_idx * out_w + out_w_idx] = min_val;
}

void fused_conv_scale_min_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    float scale_factor,
    torch::Tensor output_temp,
    torch::Tensor output) {
    
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    const int out_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    // Launch first kernel for convolution + scaling
    const int total_threads_conv = batch_size * out_channels * out_h * out_w;
    const int threads_per_block = 256;
    const int blocks_conv = (total_threads_conv + threads_per_block - 1) / threads_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_conv_scale_min_forward", ([&] {
        fused_conv_scale_min_kernel<scalar_t><<<blocks_conv, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            height,
            width,
            kernel_size,
            stride,
            padding,
            static_cast<scalar_t>(scale_factor),
            output_temp.data_ptr<scalar_t>()
        );
    }));
    
    // Launch second kernel for min reduction
    const int total_threads_reduce = batch_size * out_h * out_w;
    const int blocks_reduce = (total_threads_reduce + threads_per_block - 1) / threads_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "min_reduction_forward", ([&] {
        min_reduction_kernel<scalar_t><<<blocks_reduce, threads_per_block>>>(
            output_temp.data_ptr<scalar_t>(),
            batch_size,
            out_channels,
            out_h,
            out_w,
            output.data_ptr<scalar_t>()
        );
    }));
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_scale_min_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    float scale_factor,
    torch::Tensor output_temp,
    torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_scale_min_forward", &fused_conv_scale_min_forward, "Fused Convolution, Scaling and Min Reduction Forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_scale_min',
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
    # We only support dilation=1 and groups=1 for this implementation
    assert conv_dilation == 1, "Only dilation=1 is supported"
    assert conv_groups == 1, "Only groups=1 is supported"
    
    batch_size = x.size(0)
    in_channels = x.size(1)
    height = x.size(2)
    width = x.size(3)
    out_channels = conv_weight.size(0)
    kernel_size = conv_weight.size(2)
    
    # Calculate output dimensions
    out_h = (height + 2 * conv_padding - kernel_size) // conv_stride + 1
    out_w = (width + 2 * conv_padding - kernel_size) // conv_stride + 1
    
    # Create temporary tensor to hold conv + scale results before reduction
    temp_output = torch.empty(batch_size, out_channels, out_h, out_w, device=x.device, dtype=x.dtype)
    
    # Create final output tensor with keepdim=True (min over channel dimension)
    final_output = torch.empty(batch_size, 1, out_h, out_w, device=x.device, dtype=x.dtype)
    
    # Call our fused CUDA kernel
    fused_ext.fused_conv_scale_min_forward(
        x.contiguous(),
        conv_weight.contiguous(),
        conv_bias.contiguous(),
        conv_stride,
        conv_padding,
        float(scale_factor),
        temp_output,
        final_output
    )
    
    return final_output

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
