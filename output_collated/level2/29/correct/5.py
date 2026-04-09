# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_103731/code_2.py
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

# CUDA kernel source
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Fused kernel: Linear + Mish + Mish
// x: [batch_size, in_features]
// weight: [out_features, in_features]
// bias: [out_features]
// output: [batch_size, out_features]

__device__ __forceinline__ float mish_activation(float x) {
    // Using fast math functions for better performance
    return x * tanhf(logf(1.0f + expf(x)));
}

__global__ void fused_linear_mish_mish_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    // Each thread block handles one output feature for all batches
    int out_idx = blockIdx.x;
    if (out_idx >= out_features) return;
    
    // Shared memory for partial sums
    extern __shared__ float sdata[];
    
    for (int batch_idx = blockIdx.y; batch_idx < batch_size; batch_idx += gridDim.y) {
        float sum = 0.0f;
        
        // Each thread processes multiple input elements
        for (int i = threadIdx.x; i < in_features; i += blockDim.x) {
            int x_idx = batch_idx * in_features + i;
            int w_idx = out_idx * in_features + i;
            sum += x[x_idx] * weight[w_idx];
        }
        
        // Store partial sum in shared memory
        sdata[threadIdx.x] = sum;
        __syncthreads();
        
        // Reduction in shared memory
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }
            __syncthreads();
        }
        
        // Thread 0 has the final sum for this block
        if (threadIdx.x == 0) {
            float result = sdata[0] + bias[out_idx];
            result = mish_activation(result);
            result = mish_activation(result);
            output[batch_idx * out_features + out_idx] = result;
        }
        __syncthreads();
    }
}

// Wrapper function for kernel launch
void fused_linear_mish_mish_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output
) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);
    
    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    // Grid: (out_features, ceil(batch_size / max_blocks_per_sm))
    // We'll use a fixed number of blocks for batching
    dim3 grid(out_features, min(batch_size, 65535));
    dim3 block(256); // Block size for good occupancy
    
    // Shared memory size for reduction
    int shared_mem_size = block.x * sizeof(float);
    
    fused_linear_mish_mish_kernel<<<grid, block, shared_mem_size>>>(
        x_ptr, weight_ptr, bias_ptr, output_ptr,
        batch_size, in_features, out_features
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel failed: ", cudaGetErrorString(err));
    }
}
"""

# C++ binding source
cpp_source = r"""
#include <torch/extension.h>

void fused_linear_mish_mish_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_mish_mish", &fused_linear_mish_mish_forward, 
          "Fused linear + mish + mish kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_linear_mish_mish',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False
)

def functional_model(
    x,
    *,
    linear_weight,
    linear_bias,
):
    batch_size = x.size(0)
    out_features = linear_weight.size(0)
    
    # Create output tensor
    output = torch.empty(batch_size, out_features, dtype=x.dtype, device=x.device)
    
    # Call fused kernel
    fused_ext.fused_linear_mish_mish(x, linear_weight, linear_bias, output)
    
    return output


# Benchmark setup
batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features)]
