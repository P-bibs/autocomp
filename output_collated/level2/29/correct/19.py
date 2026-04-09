# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_105411/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['linear_weight', 'linear_bias']
REQUIRED_FLAT_STATE_NAMES = ['linear_weight', 'linear_bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Mish, and applies Mish again.
    """

    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

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
    # State for linear (nn.Linear)
    if 'linear_weight' in flat_state:
        state_kwargs['linear_weight'] = flat_state['linear_weight']
    else:
        state_kwargs['linear_weight'] = getattr(model.linear, 'weight', None)
    if 'linear_bias' in flat_state:
        state_kwargs['linear_bias'] = flat_state['linear_bias']
    else:
        state_kwargs['linear_bias'] = getattr(model.linear, 'bias', None)
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

# CUDA kernel for fused linear + double mish operation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float fast_mish(float x) {
    // Fast approximation of mish: x * tanh(softplus(x))
    // Using identity: tanh(x) ≈ x * (27.0f + x*x) / (27.0f + 9.0f*x*x) for |x| < 3.0
    float sp = x > 10.0f ? x : logf(1.0f + expf(x)); // Softplus
    float tanh_sp = tanhf(sp); // Tanh of softplus
    return x * tanh_sp;
}

__global__ void fused_linear_double_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_features,
    const int out_features
) {
    const int batch_idx = blockIdx.x;
    const int out_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || out_idx >= out_features) return;
    
    const int in_per_thread = (in_features + blockDim.y - 1) / blockDim.y;
    const int in_start = threadIdx.y * in_per_thread;
    const int in_end = min(in_start + in_per_thread, in_features);
    
    float sum = 0.0f;
    for (int k = in_start; k < in_end; ++k) {
        sum += input[batch_idx * in_features + k] * weight[out_idx * in_features + k];
    }
    
    // Reduction within warp to get partial sum
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    
    // First thread in each warp writes its result to shared memory
    __shared__ float shared_results[32]; // Assuming max 32 warps per block
    const int warp_id = threadIdx.y;
    if ((threadIdx.y < 32) && (threadIdx.y * 32 + threadIdx.x < blockDim.y)) {
        shared_results[warp_id] = (threadIdx.x == 0) ? sum : 0.0f;
    }
    __syncthreads();
    
    // Final reduction by first few threads
    if (threadIdx.y == 0 && threadIdx.x < 32 && threadIdx.x < (blockDim.y + 31)/32) {
        sum = shared_results[threadIdx.x];
    }
    __syncthreads();
    
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    
    // Add bias and apply activation
    if (threadIdx.y == 0 && threadIdx.x == 0) {
        sum += bias[out_idx];
        sum = fast_mish(sum);
        sum = fast_mish(sum);
        output[batch_idx * out_features + out_idx] = sum;
    }
}

// Optimized version using shared memory tiling
__global__ void fused_linear_double_mish_kernel_opt(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_features,
    const int out_features
) {
    const int TILE_SIZE = 32;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int batch_idx = blockIdx.x;
    const int out_tile_start = blockIdx.y * TILE_SIZE;
    
    if (batch_idx >= batch_size) return;
    
    __shared__ float shared_input[TILE_SIZE];
    __shared__ float shared_weight[TILE_SIZE * TILE_SIZE];
    
    float sum = 0.0f;
    
    // Tiled matrix multiplication
    for (int t = 0; t < (in_features + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load input tile
        int in_idx = t * TILE_SIZE + tx;
        shared_input[tx] = (in_idx < in_features) ? input[batch_idx * in_features + in_idx] : 0.0f;
        
        // Load weight tile
        int out_idx = out_tile_start + ty;
        in_idx = t * TILE_SIZE + tx;
        if (out_idx < out_features && in_idx < in_features) {
            shared_weight[ty * TILE_SIZE + tx] = weight[out_idx * in_features + in_idx];
        } else {
            shared_weight[ty * TILE_SIZE + tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += shared_input[k] * shared_weight[ty * TILE_SIZE + k];
        }
        
        __syncthreads();
    }
    
    // Write result
    int out_idx = out_tile_start + ty;
    if (out_idx < out_features) {
        sum += bias[out_idx];
        sum = fast_mish(sum);
        sum = fast_mish(sum);
        output[batch_idx * out_features + out_idx] = sum;
    }
}

void fused_linear_double_mish_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int batch_size,
    const int in_features,
    const int out_features
) {
    const dim3 block(32, 32);
    const dim3 grid(batch_size, (out_features + block.y - 1) / block.y);
    
    fused_linear_double_mish_kernel_opt<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
}
"""

# C++ interface/binding
cpp_source = r"""
#include <torch/extension.h>

void fused_linear_double_mish_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int batch_size,
    const int in_features,
    const int out_features
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_double_mish", &fused_linear_double_mish_forward, "Fused Linear + Double Mish operation");
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
    linear_weight,
    linear_bias,
):
    batch_size = x.size(0)
    in_features = x.size(1)
    out_features = linear_weight.size(0)
    
    # Create output tensor
    output = torch.empty(batch_size, out_features, device=x.device, dtype=x.dtype)
    
    # Call the fused CUDA kernel
    fused_ext.fused_linear_double_mish(
        x, linear_weight, linear_bias, output,
        batch_size, in_features, out_features
    )
    
    return output

# Helper functions (not used in eval but keeping for completeness)
batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features)]
