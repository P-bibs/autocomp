# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_190701/code_4.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a ReLU activation.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

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

# -------------------------------------------------------------------------
# Optimized CUDA source – improved vectorization with float4 and better memory coalescing
# -------------------------------------------------------------------------

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Improved fused ReLU kernel using vectorized float4 operations
__global__ void relu_fused_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   size_t n) {
    // Each thread processes 4 elements at a time using float4
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t vec_idx = tid;                // Index into the vector array
    size_t elem_base = vec_idx * 4;      // Base element index in the original array
    
    // Process full float4 vectors when possible
    if (elem_base + 3 < n) {
        // Load a float4 vector from global memory
        float4 in_vec = reinterpret_cast<const float4*>(input)[vec_idx];
        
        // Apply ReLU to each component
        in_vec.x = fmaxf(in_vec.x, 0.0f);
        in_vec.y = fmaxf(in_vec.y, 0.0f);
        in_vec.z = fmaxf(in_vec.z, 0.0f);
        in_vec.w = fmaxf(in_vec.w, 0.0f);
        
        // Store the result back as a float4 vector
        reinterpret_cast<float4*>(output)[vec_idx] = in_vec;
    } 
    // Handle the tail elements that don't fit into a full float4 vector
    else if (elem_base < n) {
        // Process remaining elements one by one
        for (size_t i = 0; i < 4 && elem_base + i < n; ++i) {
            output[elem_base + i] = fmaxf(input[elem_base + i], 0.0f);
        }
    }
}

// Host launcher with optimized grid/block dimensions
void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    size_t n = input.numel();
    
    // Use 256 threads per block for better occupancy on modern GPUs
    const int threads_per_block = 256;
    
    // Calculate number of vectorized elements (4 floats per vector)
    const size_t num_vectors = (n + 3) / 4;  // Ceiling division
    
    // Calculate number of blocks needed
    const int blocks = (num_vectors + threads_per_block - 1) / threads_per_block;
    
    // Launch the kernel
    relu_fused_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        n
    );
}
"""

# -------------------------------------------------------------------------
# C++ binding (pybind11) for Python interface
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of CUDA kernel launcher
void fused_op_forward(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Optimized fused ReLU kernel with float4 vectorization");
}
"""

# -------------------------------------------------------------------------
# Compile the inline CUDA extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional wrapper used by the benchmark harness
# -------------------------------------------------------------------------
def functional_model(x: torch.Tensor) -> torch.Tensor:
    """Applies ReLU on `x` using the optimized vectorized implementation."""
    output = torch.empty_like(x)         # Allocate output tensor
    fused_ext.fused_op(x, output)        # Launch the optimized kernel
    return output

# -------------------------------------------------------------------------
# Benchmark-related definitions (kept for reference)
# -------------------------------------------------------------------------
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]
