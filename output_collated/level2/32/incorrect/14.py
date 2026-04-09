# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_013320/code_2.py
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

# --- CUDA Kernel Code ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_conv_scale_min_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
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
    const float scale_factor
) {
    // Calculate output dimensions
    const int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Thread and block indices
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = batch_size * out_channels * out_height * out_width;
    
    if (tid >= total_threads) return;
    
    // Decompose linear index
    const int w_out = tid % out_width;
    const int h_out = (tid / out_width) % out_height;
    const int c_out = (tid / (out_width * out_height)) % out_channels;
    const int n = tid / (out_width * out_height * out_channels);
    
    // Calculate input coordinates
    const int h_in_start = h_out * stride - padding;
    const int w_in_start = w_out * stride - padding;
    
    // Perform convolution for this output element
    float conv_result = 0.0f;
    
    // Loop over input channels (grouped)
    const int group_idx = c_out / (out_channels / groups);
    const int channels_per_group = in_channels / groups;
    const int weight_offset_base = c_out * channels_per_group * kernel_size * kernel_size;
    
    for (int c_in_group = 0; c_in_group < channels_per_group; ++c_in_group) {
        const int c_in = group_idx * channels_per_group + c_in_group;
        const int input_batch_offset = (n * in_channels + c_in) * height * width;
        const int weight_offset = weight_offset_base + c_in_group * kernel_size * kernel_size;
        
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int h_in = h_in_start + kh * dilation;
                const int w_in = w_in_start + kw * dilation;
                
                // Check bounds
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    const float val = input[input_batch_offset + h_in * width + w_in];
                    const float wgt = weight[weight_offset + kh * kernel_size + kw];
                    conv_result += val * wgt;
                }
            }
        }
    }
    
    // Add bias
    conv_result += bias[c_out];
    
    // Scale the result
    conv_result *= scale_factor;
    
    // Since we're doing min across channels, we need to compute all channels for this spatial location
    // This approach works but is inefficient. A better approach would use shared memory and reduction.
    // For now, we'll compute the min in a separate step to maintain correctness.
    
    // Store intermediate result
    const int output_idx = ((n * out_channels + c_out) * out_height + h_out) * out_width + w_out;
    output[output_idx] = conv_result;
}

__global__ void min_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int out_channels,
    const int out_height,
    const int out_width
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_height * out_width;
    
    if (tid >= total_elements) return;
    
    const int w = tid % out_width;
    const int h = (tid / out_width) % out_height;
    const int n = tid / (out_width * out_height);
    
    float min_val = INFINITY;
    for (int c = 0; c < out_channels; ++c) {
        const int idx = ((n * out_channels + c) * out_height + h) * out_width + w;
        min_val = fminf(min_val, input[idx]);
    }
    
    const int output_idx = (n * out_height + h) * out_width + w;
    output[output_idx] = min_val;
}

void fused_conv_scale_min_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output_intermediate,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups,
    float scale_factor
) {
    // Ensure tensors are on CUDA
    at:: cuda::CUDAGuard device_guard(input.device().index());
    
    const int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    const int total_threads_conv = batch_size * out_channels * out_height * out_width;
    const int threads_per_block = 256;
    const int blocks_conv = (total_threads_conv + threads_per_block - 1) / threads_per_block;
    
    fused_conv_scale_min_kernel<<<blocks_conv, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output_intermediate.data_ptr<float>(),
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
        scale_factor
    );
    
    const int total_threads_reduce = batch_size * out_height * out_width;
    const int blocks_reduce = (total_threads_reduce + threads_per_block - 1) / threads_per_block;
    
    min_reduce_kernel<<<blocks_reduce, threads_per_block>>>(
        output_intermediate.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_channels,
        out_height,
        out_width
    );
    
    cudaDeviceSynchronize();
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_scale_min_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output_intermediate,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups,
    float scale_factor
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_scale_min_forward", &fused_conv_scale_min_forward, "Fused Conv + Scale + Min Reduction Forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
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
    batch_size, in_channels, height, width = x.shape
    out_channels = conv_weight.shape[0]
    kernel_size = conv_weight.shape[2]
    
    # Calculate output dimensions
    out_height = (height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_width = (width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    # Create intermediate tensor for conv + scale output
    output_intermediate = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Create final output tensor
    output = torch.empty(batch_size, 1, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_conv_scale_min_forward(
        x, conv_weight, conv_bias,
        output_intermediate, output,
        batch_size, in_channels, out_channels,
        height, width, kernel_size,
        conv_stride, conv_padding, conv_dilation, conv_groups,
        scale_factor
    )
    
    return output

# Test parameters
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
