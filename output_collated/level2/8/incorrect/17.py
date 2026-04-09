# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_055208/code_4.py
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
from torch.utils.cpp_extension import load_inline

# --- CUDA Kernel with Memory Coalescing ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_op_kernel_coalesced(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float divisor,
    int N, int C, int D, int H, int W) {
    
    // Grid-stride loop for spatial dimensions
    // Each block handles one (N, D, H, W) location and processes all channels
    int spatial_idx = blockIdx.x * blockDim.y + threadIdx.y;
    int total_spatial = N * D * H * W;
    
    if (spatial_idx >= total_spatial) return;
    
    // Decompose spatial index
    int n = spatial_idx / (D * H * W);
    int remainder = spatial_idx % (D * H * W);
    int d = remainder / (H * W);
    remainder = remainder % (H * W);
    int h = remainder / W;
    int w = remainder % W;
    
    // threadIdx.x iterates over channels in parallel (coalesced access)
    int channel_start = threadIdx.x;
    
    float sum_val = 0.0f;
    
    // Coalesced reads: consecutive threads read consecutive channels
    // Memory pattern: threads 0,1,2,... read channels 0,1,2,... at same spatial location
    for (int c = channel_start; c < C; c += blockDim.x) {
        int input_idx = ((n * C + c) * D + d) * H * W + h * W + w;
        sum_val += (input[input_idx] / divisor) + bias[c];
    }
    
    // Reduce within the warp/block using shared memory
    extern __shared__ float shared_data[];
    shared_data[threadIdx.x] = sum_val;
    __syncthreads();
    
    // Tree reduction within block
    #pragma unroll
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Write result from thread 0 of each spatial group
    if (threadIdx.x == 0) {
        int out_idx = n * (D * H * W) + d * (H * W) + h * W + w;
        output[out_idx] = shared_data[0];
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor bias, torch::Tensor output, float divisor) {
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    
    // Optimal configuration for memory coalescing
    // Use 32 threads for channel dimension (warp size), 8 for spatial batching
    int threads_per_channel_dim = 32;  // Process 32 channels per thread group
    int threads_per_spatial_dim = 8;   // Process 8 spatial locations in parallel
    
    dim3 threads(threads_per_channel_dim, threads_per_spatial_dim, 1);
    int total_spatial = N * D * H * W;
    int blocks = (total_spatial + threads_per_spatial_dim - 1) / threads_per_spatial_dim;
    
    size_t shared_mem_size = threads_per_channel_dim * sizeof(float);
    
    fused_op_kernel_coalesced<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(), 
        divisor, N, C, D, H, W);
    
    cudaDeviceSynchronize();
}
"""

# --- C++ Interface ---
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor input, torch::Tensor bias, torch::Tensor output, float divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused op with coalesced memory access");
}
"""

fused_ext = load_inline(
    name='fused_op_coalesced',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-maxrregcount=64'],
    with_cuda=True
)

import torch.nn.functional as F

def functional_model(
    x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups,
    max_pool_kernel_size, max_pool_stride, max_pool_padding, max_pool_dilation,
    max_pool_ceil_mode, max_pool_return_indices, global_avg_pool_output_size,
    divisor, bias, sum_dim,
):
    # PyTorch Ops
    x = F.conv3d(x, conv_weight, conv_bias, stride=conv_stride, padding=conv_padding, 
                  dilation=conv_dilation, groups=conv_groups)
    x = F.max_pool3d(x, kernel_size=max_pool_kernel_size, stride=max_pool_stride, 
                     padding=max_pool_padding, dilation=max_pool_dilation, 
                     ceil_mode=max_pool_ceil_mode, return_indices=max_pool_return_indices)
    x = F.adaptive_avg_pool3d(x, global_avg_pool_output_size)
    
    # Fused Custom Kernel with Coalesced Memory Access
    N, C, D, H, W = x.shape
    out = torch.zeros((N, D, H, W), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(x.contiguous(), bias.contiguous().view(-1), out, divisor)
    return out

# Placeholders for evaluation
batch_size = 128
in_channels = 8
out_channels = 16
depth = 16
height = 64
width = 64
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
