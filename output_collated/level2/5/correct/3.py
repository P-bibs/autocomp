# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_115905/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'bias_shape', 'stride', 'padding', 'output_padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

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
    # State for conv_transpose (nn.ConvTranspose2d)
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
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
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
from torch.utils.cpp_extension import load_inline

# CUDA Kernel: Optimized with shared memory caching and coalesced access
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_op_kernel_optimized(
    float* __restrict__ data, 
    const float* __restrict__ bias, 
    int N, int C, int HW
) {
    int total_elements = N * C * HW;
    
    // Shared memory for bias caching
    extern __shared__ float shared_bias[];
    
    // Cooperatively load bias into shared memory
    int bias_load_idx = threadIdx.x;
    while (bias_load_idx < C) {
        shared_bias[bias_load_idx] = bias[bias_load_idx];
        bias_load_idx += blockDim.x;
    }
    __syncthreads();
    
    // Process data with coalesced memory access
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Precompute constants to avoid expensive division
    int C_HW = C * HW;
    
    while (idx < total_elements) {
        // Efficient index computation using precomputed constants
        int n = idx / C_HW;
        int remainder = idx % C_HW;
        int c = remainder / HW;
        int hw = remainder % HW;
        
        float val = data[idx];
        val = tanhf(val - shared_bias[c]);
        data[idx] = val;
        
        idx += gridDim.x * blockDim.x;
    }
}

void fused_op_forward(torch::Tensor& x, const torch::Tensor& bias) {
    int N = x.size(0);
    int C = x.size(1);
    int HW = x.size(2) * x.size(3);
    int total_elements = N * C * HW;
    
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    // Limit blocks to avoid excessive kernel launches
    blocks = std::min(blocks, 65536);
    
    size_t shared_mem_size = C * sizeof(float);
    
    fused_op_kernel_optimized<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        N, C, HW
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor& x, const torch::Tensor& bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op_forward", &fused_op_forward, "Fused bias subtraction and tanh with optimized memory access");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
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
    bias,
):
    # Perform convolution (using PyTorch's optimized backend)
    x = torch.conv_transpose2d(
        x, conv_transpose_weight, conv_transpose_bias, 
        stride=conv_transpose_stride, 
        padding=conv_transpose_padding, 
        output_padding=conv_transpose_output_padding, 
        groups=conv_transpose_groups, 
        dilation=conv_transpose_dilation
    )
    
    # Flatten bias for kernel usage
    bias_flat = bias.view(-1)
    
    # Run optimized kernel with shared memory caching
    fused_ext.fused_op_forward(x, bias_flat)
    
    return x

# Setup for testing
batch_size = 32
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]
