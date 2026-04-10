# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_125421/code_14.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['dim']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs sum reduction over a specified dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

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
    if 'dim' in flat_state:
        state_kwargs['dim'] = flat_state['dim']
    else:
        state_kwargs['dim'] = getattr(model, 'dim')
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
#  Optimized CUDA implementation with shared memory reduction
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tile size for shared memory reduction - must be power of 2 for tree reduction
#define TILE_SIZE 256

__global__ void sum_dim1_kernel_opt(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int B, int D1, int D2) {
    // Shared memory for partial sums
    __shared__ float sdata[TILE_SIZE];
    
    // Each block handles one output element (b, j)
    int j = blockIdx.x;
    int b = blockIdx.y;
    
    if (b >= B || j >= D2) return;
    
    // Calculate base pointer for this batch and output position
    const float* __restrict__ b_ptr = input + (b * D1 * D2) + j;
    
    int tid = threadIdx.x;
    
    // Cooperative loading: each thread loads a portion of D1 elements
    // Threads access D2-strided data (coalesced for fixed j)
    float psum = 0.0f;
    
    // Process elements with grid-stride loop
    for (int i = tid; i < D1; i += blockDim.x) {
        psum += b_ptr[i * D2];
    }
    
    // Store partial sum to shared memory
    sdata[tid] = psum;
    __syncthreads();
    
    // Parallel reduction in shared memory (tree reduction)
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Final warp-level reduction (no sync needed within warp)
    if (tid < 32) {
        volatile float* vsdata = sdata; // Prevent compiler optimization
        vsdata[tid] += vsdata[tid + 32];
        vsdata[tid] += vsdata[tid + 16];
        vsdata[tid] += vsdata[tid + 8];
        vsdata[tid] += vsdata[tid + 4];
        vsdata[tid] += vsdata[tid + 2];
        vsdata[tid] += vsdata[tid + 1];
    }
    
    // Thread 0 writes the final result
    if (tid == 0) {
        output[b * D2 + j] = sdata[0];
    }
}

void sum_dim1_gpu_opt(torch::Tensor input, torch::Tensor output) {
    int B = input.size(0);
    int D1 = input.size(1);
    int D2 = input.size(2);
    
    // Use 256 threads per block for optimal reduction
    const int threads = 256;
    
    // Grid: D2 blocks for the D2 dimension, B blocks for batch
    dim3 block(threads);
    dim3 grid(D2, B);
    
    sum_dim1_kernel_opt<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void sum_dim1_gpu_opt(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1_gpu_opt", &sum_dim1_gpu_opt, "Optimized sum along dim 1 with shared memory reduction");
}
"""

# Compile the optimized extension
sum_ext_opt = load_inline(
    name='sum_dim1_opt_v2',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    """
    Sum along dimension 1 using custom optimized CUDA kernel with shared memory reduction.
    Shape: (B, D1, D2) -> (B, 1, D2)
    """
    assert dim == 1, "Only dim=1 is supported."
    
    batch, d1, d2 = x.shape
    output = torch.zeros((batch, d2), device=x.device, dtype=x.dtype)
    
    # Kernel computes (B, D2) result with cooperative reduction
    sum_ext_opt.sum_dim1_gpu_opt(x, output)
    
    return output.view(batch, 1, d2)

if __name__ == "__main__":
    # Sanity check
    batch_size, d1, d2 = 32, 128, 64
    x = torch.randn(batch_size, d1, d2, device='cuda')
    expected = x.sum(dim=1, keepdim=True)
    actual = functional_model(x, dim=1)
    
    assert torch.allclose(actual, expected, atol=1e-5)
    print("Optimization successful: output matches torch.sum.")
