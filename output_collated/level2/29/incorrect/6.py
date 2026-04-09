# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110005/code_2.py
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused linear + 2x mish operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32
#define MAX_BLOCK_SIZE 512

__device__ __forceinline__ float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__device__ __forceinline__ float mish(float x) {
    return x * tanhf(softplus(x));
}

__global__ void fused_linear_2xmish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = blockDim.x;
    
    // Each block processes one output row
    if (bid >= batch_size) return;
    
    // Shared memory for caching input and weight
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + in_features;
    
    // Load input into shared memory
    for (int i = tid; i < in_features; i += block_size) {
        shared_input[i] = input[bid * in_features + i];
    }
    
    // Process each output feature
    for (int out_idx = 0; out_idx < out_features; out_idx++) {
        // Load weight row into shared memory
        for (int i = tid; i < in_features; i += block_size) {
            shared_weight[i] = weight[out_idx * in_features + i];
        }
        __syncthreads();
        
        // Compute linear output
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            sum += shared_input[i] * shared_weight[i];
        }
        
        // Add bias
        sum += bias[out_idx];
        
        // Apply first Mish activation
        float result = mish(sum);
        
        // Apply second Mish activation
        result = mish(result);
        
        // Write result
        output[bid * out_features + out_idx] = result;
        
        __syncthreads();
    }
}

void fused_linear_2xmish_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_features,
    int out_features
) {
    // Determine grid and block dimensions
    int block_size = min(MAX_BLOCK_SIZE, in_features);
    block_size = (block_size + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE; // Round to warp size
    int grid_size = batch_size;
    
    // Calculate shared memory size (input + one row of weight)
    size_t shared_mem_size = (in_features + in_features) * sizeof(float);
    
    fused_linear_2xmish_kernel<<<grid_size, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ source for binding
cpp_source = r"""
#include <torch/extension.h>

void fused_linear_2xmish_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_features,
    int out_features
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_2xmish_forward", &fused_linear_2xmish_forward, "Fused Linear + 2x Mish forward pass");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_linear_2xmish',
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
    
    # Call fused kernel
    fused_ext.fused_linear_2xmish_forward(
        x.contiguous(),
        linear_weight.contiguous(),
        linear_bias.contiguous(),
        output,
        batch_size,
        in_features,
        out_features
    )
    
    return output

# Test parameters
batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features)]
