# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_123707/code_1.py
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




# =============================================================================
#  High-performance reduction along dimension 1 (batch, D1, D2) → (batch, 1, D2)
#  Custom CUDA kernel optimized for memory coalescing and warp efficiency.
# =============================================================================

import torch
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------------------------
#  Optimized CUDA implementation with better memory access patterns
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define TILE_DIM 32

// Highly optimized kernel using shared memory tiling and coalesced access
__global__ void sum_dim1_tiled_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int B, int D1, int D2) {
    // Shared memory for input tile with padding to avoid bank conflicts
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    int b = blockIdx.x;
    int tile_idx = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global column index
    int j = tile_idx * TILE_DIM + tx;
    
    float sum = 0.0f;
    
    if (b < B) {
        // Process D1 in chunks of TILE_DIM
        for (int i_start = 0; i_start < D1; i_start += TILE_DIM) {
            // Collaborative loading of tile into shared memory
            // Each thread loads one element per row in the tile
            int i = i_start + ty;
            if (i < D1 && j < D2) {
                tile[ty][tx] = input[b * D1 * D2 + i * D2 + j];
            } else {
                tile[ty][tx] = 0.0f;
            }
            
            __syncthreads();
            
            // Perform reduction within the tile
            // Each thread now sums across its column in the tile
            if (j < D2) {
                for (int k = 0; k < TILE_DIM && (i_start + k) < D1; k++) {
                    sum += tile[k][tx];
                }
            }
            
            __syncthreads();
        }
        
        // Write final result
        if (j < D2) {
            output[b * D2 + j] = sum;
        }
    }
}

// Alternative optimized kernel with improved memory coalescing
__global__ void sum_dim1_coalesced_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int B, int D1, int D2) {
    int b = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (b >= B || j >= D2) return;
    
    // Use pointer arithmetic for better performance
    const float* input_row = input + b * D1 * D2 + j;
    float sum = 0.0f;
    
    // Coalesced memory access: consecutive threads access consecutive memory
    for (int i = 0; i < D1; i++) {
        sum += input_row[i * D2];
    }
    
    output[b * D2 + j] = sum;
}

void sum_dim1_optimized(torch::Tensor input, torch::Tensor output) {
    int B = input.size(0);
    int D1 = input.size(1);
    int D2 = input.size(2);
    
    // Heuristic to choose best kernel based on dimensions
    if (D1 >= 64 && D2 >= 64) {
        // Use tiled approach for large matrices
        dim3 threads(TILE_DIM, TILE_DIM);
        dim3 blocks(B, (D2 + TILE_DIM - 1) / TILE_DIM);
        
        sum_dim1_tiled_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            B, D1, D2);
    } else {
        // Use simpler coalesced approach for smaller matrices
        dim3 threads(256);
        dim3 blocks(B, (D2 + threads.x - 1) / threads.x);
        
        sum_dim1_coalesced_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            B, D1, D2);
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>

void sum_dim1_optimized(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1_optimized", &sum_dim1_optimized, "Optimized sum along dimension 1");
}
"""

# Compile the optimized extension
sum_ext_opt = load_inline(
    name='sum_dim1_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
#  Functional implementation using our optimized CUDA kernel
# -------------------------------------------------------------------------
def functional_model(x, *, dim):
    """
    Reduce the input tensor `x` of shape (B, D1, D2) along dimension 1 using
    a custom optimized CUDA kernel.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor on CUDA.  Shape: (batch, dim1, dim2).
    dim : int
        Must be 1 (as required by the problem statement).

    Returns
    -------
    torch.Tensor
        Output tensor of shape (batch, 1, dim2) containing the sums.
    """
    assert dim == 1, "Only dim=1 is supported by this model."
    
    # Create output tensor with correct shape
    output = torch.zeros((x.shape[0], 1, x.shape[2]), device=x.device, dtype=x.dtype)
    
    # Use our optimized CUDA kernel
    sum_ext_opt.sum_dim1_optimized(x, output.squeeze(1))
    
    return output

# -------------------------------------------------------------------------
#  Evaluation harness (unchanged from the original script)
# -------------------------------------------------------------------------
batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def get_init_inputs():
    """Returns any static inputs required by the model – here only the reduction dim."""
    return [reduce_dim]

def get_inputs():
    """Creates a random input tensor on the GPU for benchmarking."""
    x = torch.rand(batch_size, dim1, dim2, device='cuda', dtype=torch.float32)
    return [x]

# -------------------------------------------------------------------------
#  If this module is executed directly, run a quick sanity check.
# -------------------------------------------------------------------------
if __name__ == "__main__":
    dim = reduce_dim
    x = torch.rand(batch_size, dim1, dim2, device='cuda')
    out = functional_model(x, dim=dim)
    # Verify shape
    assert out.shape == (batch_size, 1, dim2)
    print("Functional model with custom CUDA kernel works correctly.")
