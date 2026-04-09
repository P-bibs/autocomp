# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_051902/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies HardSwish, and then ReLU.
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused convolution + hardswish + relu
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__forceinline__ __device__ float hardswish_impl(float x) {
    float relu6_val = fmaxf(0.0f, fminf(x + 3.0f, 6.0f));
    return x * relu6_val / 6.0f;
}

__forceinline__ __device__ float relu_impl(float x) {
    return fmaxf(0.0f, x);
}

__global__ void fused_conv_activation_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    // Calculate output dimensions
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Get thread indices
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * out_channels * out_height * out_width;
    
    if (tid >= total_threads) return;
    
    // Decompose linear thread index
    int batch_idx = tid / (out_channels * out_height * out_width);
    int remainder = tid % (out_channels * out_height * out_width);
    int out_ch = remainder / (out_height * out_width);
    remainder = remainder % (out_height * out_width);
    int out_y = remainder / out_width;
    int out_x = remainder % out_width;
    
    // Calculate weight group index
    int group_idx = out_ch * groups / out_channels;
    int in_channels_per_group = in_channels / groups;
    
    float sum = 0.0f;
    
    // Convolution computation
    for (int in_ch = group_idx * in_channels_per_group; in_ch < (group_idx + 1) * in_channels_per_group; ++in_ch) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_y = out_y * stride - padding + ky * dilation;
                int in_x = out_x * stride - padding + kx * dilation;
                
                if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                    int input_idx = batch_idx * (in_channels * height * width) + 
                                   in_ch * (height * width) + 
                                   in_y * width + in_x;
                                   
                    int weight_idx = out_ch * (in_channels_per_group * kernel_size * kernel_size) +
                                    (in_ch - group_idx * in_channels_per_group) * (kernel_size * kernel_size) +
                                    ky * kernel_size + kx;
                                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    sum += bias[out_ch];
    
    // Apply hardswish then relu
    sum = hardswish_impl(sum);
    sum = relu_impl(sum);
    
    // Write output
    output[tid] = sum;
}

void launch_fused_conv_activation(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
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
    cudaStream_t stream) {
    
    // Calculate output dimensions
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Configure kernel launch parameters
    int total_threads = batch_size * out_channels * out_height * out_width;
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;
    
    fused_conv_activation_kernel<<<grid_size, block_size, 0, stream>>>(
        input, weight, bias, output,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, stride, padding, dilation, groups
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void launch_fused_conv_activation(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
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
    cudaStream_t stream);

torch::Tensor fused_conv_activation(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    // Ensure tensors are on CUDA
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(bias.is_cuda(), "Bias tensor must be on CUDA");
    
    // Ensure tensor types
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    // Calculate output dimensions
    auto out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Create output tensor
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, 
                              torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Set CUDA device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    // Launch kernel
    launch_fused_conv_activation(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
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
        at::cuda::getCurrentCUDAStream()
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_activation", &fused_conv_activation, "Fused convolution with hardswish and relu activation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_activation_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
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
):
    # Use our fused kernel for convolution + activations
    return fused_ext.fused_conv_activation(
        x, conv_weight, conv_bias, 
        conv_stride, conv_padding, conv_dilation, conv_groups
    )

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
