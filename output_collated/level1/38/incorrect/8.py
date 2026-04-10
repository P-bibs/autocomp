# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_022930/code_15.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs L1 normalization.
    """

    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
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

# CUDA kernel for fused operation: abs + mean + division with coalesced memory access
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void fused_norm_kernel(const float* __restrict__ x, float* __restrict__ out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    // Use shared memory for reduction
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Calculate chunk size per thread for coalesced access
    int elems_per_thread = (cols + block_size - 1) / block_size;
    int start = tid * elems_per_thread;
    int end = min(start + elems_per_thread, cols);
    
    const float* row_ptr = x + row * cols;
    
    // Compute sum of absolute values in a coalesced manner
    float thread_sum = 0.0f;
    for (int i = start; i < end; ++i) {
        thread_sum += fabsf(__ldg(&row_ptr[i]));
    }
    
    // Store in shared memory
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Reduce within the block using shared memory
    for (int s = block_size / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Warp-level reduction
    if (tid < 32) {
        sdata[tid] = warpReduceSum(sdata[tid]);
    }
    __syncthreads();
    
    // Only thread 0 computes the final mean
    float mean = sdata[0] / (float)cols;
    
    // Apply normalization (division by mean) with coalesced writes
    for (int i = start; i < end; ++i) {
        out[row * cols + i] = __ldg(&row_ptr[i]) / mean;
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor out) {
    // Ensure we're on the correct device
    at::cuda::CUDAGuard device_guard(x.device());
    
    int rows = x.size(0);
    int cols = x.size(1);
    int threads = 512;
    
    // Launch one block per row
    fused_norm_kernel<<<rows, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        rows, 
        cols
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the CUDA function
void fused_op_forward(torch::Tensor x, torch::Tensor out);

// Binding to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op_forward", &fused_op_forward, "Fused Norm Forward (CUDA)");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_norm_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """Optimized version using custom CUDA kernel with coalesced memory access."""
    x = x.contiguous()
    out = torch.empty_like(x)
    fused_ext.fused_op_forward(x, out)
    return out

# Testing parameters
batch_size = 32768
dim = 65535

def get_inputs():
    return [torch.rand(batch_size, dim, dtype=torch.float32, device='cuda')]
