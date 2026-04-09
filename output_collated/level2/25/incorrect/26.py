# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_083856/code_5.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused conv + min + tanh operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ conv_output,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_height,
    const int out_width
) {
    const int batch_idx = blockIdx.x;
    const int hw_idx = blockIdx.y * blockDim.x + threadIdx.x;
    const int total_hw = out_height * out_width;
    
    if (batch_idx >= batch_size || hw_idx >= total_hw) return;
    
    const int out_idx = batch_idx * total_hw + hw_idx;
    
    float min_val = INFINITY;
    const int base_idx = batch_idx * in_channels * total_hw + hw_idx;
    
    // Find minimum across channels
    for (int c = 0; c < in_channels; ++c) {
        const int idx = base_idx + c * total_hw;
        float val = conv_output[idx];
        min_val = fminf(min_val, val);
    }
    
    // Apply tanh
    output[out_idx] = tanhf(min_val);
}

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    const int out_ch = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int hw_idx = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (out_ch >= out_channels || batch_idx >= batch_size || hw_idx >= out_height * out_width) return;
    
    const int out_y = hw_idx / out_width;
    const int out_x = hw_idx % out_width;
    
    const int kernel_radius = (kernel_size - 1) / 2;
    float sum = 0.0f;
    
    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                const int in_y = out_y * stride - padding + ky * dilation;
                const int in_x = out_x * stride - padding + kx * dilation;
                
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    const int in_idx = batch_idx * in_channels * in_height * in_width +
                                      in_ch * in_height * in_width +
                                      in_y * in_width + in_x;
                    const int weight_idx = out_ch * in_channels * kernel_size * kernel_size +
                                          in_ch * kernel_size * kernel_size +
                                          ky * kernel_size + kx;
                    sum += input[in_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    const int out_idx = batch_idx * out_channels * out_height * out_width +
                       out_ch * out_height * out_width +
                       out_y * out_width + out_x;
    output[out_idx] = sum + bias[out_ch];
}

void fused_conv_min_tanh_op(
    const torch::Tensor conv_output,
    torch::Tensor output
) {
    const at::cuda::OptionalCUDAGuard device_guard(conv_output.device());
    
    const int batch_size = conv_output.size(0);
    const int in_channels = conv_output.size(1);
    const int out_height = conv_output.size(2);
    const int out_width = conv_output.size(3);
    
    const dim3 block_size(256);
    const dim3 grid_size(batch_size, (out_height * out_width + block_size.x - 1) / block_size.x);
    
    fused_conv_min_tanh_kernel<<<grid_size, block_size>>>(
        conv_output.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_height,
        out_width
    );
    
    AT_CUDA_CHECK(cudaGetLastError());
}

void custom_conv2d_op(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int stride,
    const int padding,
    const int dilation
) {
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    const dim3 block_size(256);
    const dim3 grid_size(out_channels, batch_size, (out_height * out_width + block_size.x - 1) / block_size.x);
    
    conv2d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        dilation
    );
    
    AT_CUDA_CHECK(cudaGetLastError());
}
"""

# C++ bindings
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_min_tanh_op(
    const torch::Tensor conv_output,
    torch::Tensor output
);

void custom_conv2d_op(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int stride,
    const int padding,
    const int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_min_tanh", &fused_conv_min_tanh_op, "Fused conv min tanh operation");
    m.def("custom_conv2d", &custom_conv2d_op, "Custom conv2d operation");
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
):
    # Perform convolution using custom CUDA kernel
    batch_size = x.size(0)
    in_channels = x.size(1)
    in_height = x.size(2)
    in_width = x.size(3)
    out_channels = conv_weight.size(0)
    kernel_size = conv_weight.size(2)
    
    out_height = (in_height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_width = (in_width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    conv_output = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    fused_ext.custom_conv2d(x, conv_weight, conv_bias, conv_output, conv_stride, conv_padding, conv_dilation)
    
    # Apply fused min + tanh operation
    output = torch.empty(batch_size, 1, out_height, out_width, device=x.device, dtype=x.dtype)
    fused_ext.fused_conv_min_tanh(conv_output, output)
    
    return output

batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
