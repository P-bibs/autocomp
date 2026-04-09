# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_053523/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'divisor', 'pool_size', 'bias_shape', 'sum_dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'max_pool_kernel_size', 'max_pool_stride', 'max_pool_padding', 'max_pool_dilation', 'max_pool_ceil_mode', 'max_pool_return_indices', 'global_avg_pool_output_size', 'divisor', 'bias', 'sum_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    """

    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

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
    # State for conv (nn.Conv3d)
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
    # State for max_pool (nn.MaxPool3d)
    state_kwargs['max_pool_kernel_size'] = model.max_pool.kernel_size
    state_kwargs['max_pool_stride'] = model.max_pool.stride
    state_kwargs['max_pool_padding'] = model.max_pool.padding
    state_kwargs['max_pool_dilation'] = model.max_pool.dilation
    state_kwargs['max_pool_ceil_mode'] = model.max_pool.ceil_mode
    state_kwargs['max_pool_return_indices'] = model.max_pool.return_indices
    # State for global_avg_pool (nn.AdaptiveAvgPool3d)
    state_kwargs['global_avg_pool_output_size'] = model.global_avg_pool.output_size
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
    if 'sum_dim' in flat_state:
        state_kwargs['sum_dim'] = flat_state['sum_dim']
    else:
        state_kwargs['sum_dim'] = getattr(model, 'sum_dim')
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

# CUDA kernel for fused operations: division, bias addition, and sum reduction
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_op_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int num_elements,
    const int channels,
    const int spatial_dim,
    const float divisor
) {
    // Get global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        // Calculate channel index for bias broadcasting
        int channel_idx = (idx / spatial_dim) % channels;
        
        // Perform division and bias addition
        float result = (input[idx] / divisor) + bias[channel_idx];
        
        // Write to output
        output[idx] = result;
    }
}

__global__ void sum_reduction_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int spatial_dim
) {
    // Each block handles one batch element
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Shared memory for reduction
    extern __shared__ float sdata[];
    
    // Initialize shared memory
    sdata[tid] = 0.0f;
    
    // Grid-stride loop to accumulate values
    int elements_per_batch = channels * spatial_dim;
    int start_idx = batch_idx * elements_per_batch;
    
    for (int i = tid; i < elements_per_batch; i += blockDim.x) {
        sdata[tid] += input[start_idx + i];
    }
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[batch_idx] = sdata[0];
    }
}

void fused_op_forward(
    const torch::Tensor input,
    const torch::Tensor bias,
    torch::Tensor output,
    const float divisor
) {
    const int num_elements = input.numel();
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int spatial_dim = input.size(2) * input.size(3) * input.size(4);
    
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;
    
    fused_op_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        channels,
        spatial_dim,
        divisor
    );
    
    cudaDeviceSynchronize();
}

void sum_reduction_forward(
    const torch::Tensor input,
    torch::Tensor output
) {
    const int batch_size = input.size(0);
    const int threads = 256;
    
    // Shared memory size for reduction
    const int shared_mem_size = threads * sizeof(float);
    
    sum_reduction_kernel<<<batch_size, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input.size(1), // channels
        input.size(2) * input.size(3) * input.size(4) // spatial dimensions
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ bindings for the CUDA functions
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(const torch::Tensor input, const torch::Tensor bias, torch::Tensor output, const float divisor);
void sum_reduction_forward(const torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused division and bias addition");
    m.def("sum_reduction", &sum_reduction_forward, "Sum reduction along channel dimension");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_operations',
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
    max_pool_kernel_size,
    max_pool_stride,
    max_pool_padding,
    max_pool_dilation,
    max_pool_ceil_mode,
    max_pool_return_indices,
    global_avg_pool_output_size,
    divisor,
    bias,
    sum_dim,
):
    # Conv3D operation
    x = F.conv3d(x, conv_weight, conv_bias, stride=conv_stride, padding=conv_padding, dilation=conv_dilation, groups=conv_groups)
    
    # Max pooling
    x = F.max_pool3d(x, kernel_size=max_pool_kernel_size, stride=max_pool_stride, padding=max_pool_padding, dilation=max_pool_dilation, ceil_mode=max_pool_ceil_mode, return_indices=max_pool_return_indices)
    
    # Adaptive average pooling
    x = F.adaptive_avg_pool3d(x, global_avg_pool_output_size)
    
    # Fused division and bias addition
    output = torch.empty_like(x)
    fused_ext.fused_op(x, bias, output, divisor)
    x = output
    
    # Sum reduction
    batch_size = x.size(0)
    output_sum = torch.empty(batch_size, device=x.device, dtype=x.dtype)
    fused_ext.sum_reduction(x, output_sum)
    
    return output_sum

batch_size   = 128  
in_channels  = 8            
out_channels = 16  
depth = 16; height = width = 64 
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
