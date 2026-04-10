# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_143434/code_6.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 256

__global__ void sum_dim1_shared_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       const int B, const int D1, const int D2) {
    // Shared memory for block-level reduction
    __shared__ float sdata[MAX_THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int b = blockIdx.x;
    int col = blockIdx.y * blockDim.x + tid;
    
    // Initialize shared memory
    sdata[tid] = 0.0f;
    
    // Each thread accumulates values from D1 dimension
    if (b < B && col < D2) {
        const float* input_ptr = input + b * D1 * D2 + col;
        
        // Accumulate along D1 dimension with coalesced access
        #pragma unroll 4
        for (int i = 0; i < D1; ++i) {
            sdata[tid] += input_ptr[i * D2];
        }
    }
    
    __syncthreads();
    
    // Perform block-level reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (col + s) < D2) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result to global memory
    if (tid == 0 && b < B && col < D2) {
        output[b * D2 + col] = sdata[0];
    }
    
    // Handle remaining elements that don't fit in the reduction pattern
    if (blockDim.x >= WARP_SIZE && (blockDim.x & (blockDim.x - 1)) == 0) {
        // Power of 2 block size - optimized path
        for (int s = WARP_SIZE; s < blockDim.x; s *= 2) {
            if (tid == 0 && (col + s) < D2) {
                sdata[0] += sdata[s];
            }
            __syncthreads();
        }
    }
}

// Warp-level reduction using shuffle operations for better performance
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void sum_dim1_warp_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     const int B, const int D1, const int D2) {
    int b = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    // Each thread accumulates values from D1 dimension
    if (b < B && col < D2) {
        const float* input_ptr = input + b * D1 * D2 + col;
        
        // Accumulate along D1 dimension
        #pragma unroll 4
        for (int i = 0; i < D1; ++i) {
            sum += input_ptr[i * D2];
        }
    }
    
    // Perform warp-level reduction for better performance
    sum = warpReduceSum(sum);
    
    // First thread in each warp writes result
    if (threadIdx.x % WARP_SIZE == 0 && b < B && col < D2) {
        output[b * D2 + col] = sum;
    }
}

// Final optimized version with proper grid-stride loops and vectorized loads
__global__ void sum_dim1_final_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      const int B, const int D1, const int D2) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int col = blockIdx.y * blockDim.x + tid;
    
    // Shared memory for block reduction
    __shared__ float sdata[MAX_THREADS_PER_BLOCK];
    
    float sum = 0.0f;
    
    // Grid-stride loop to handle cases where D2 > number of threads
    for (int c = col; c < D2; c += blockDim.x * gridDim.y) {
        if (b < B) {
            const float* input_ptr = input + b * D1 * D2 + c;
            
            // Accumulate along D1 dimension
            for (int i = 0; i < D1; ++i) {
                sum += input_ptr[i * D2];
            }
        }
    }
    
    // Store partial sum in shared memory
    sdata[tid] = sum;
    __syncthreads();
    
    // Perform block-level reduction
    for (int s = blockDim.x / 2; s > WARP_SIZE; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Warp-level reduction
    if (tid < WARP_SIZE) {
        sdata[tid] = warpReduceSum(sdata[tid]);
    }
    
    // Thread 0 writes the final result
    if (tid == 0 && b < B) {
        int c = blockIdx.y * blockDim.x;
        if (c < D2) {
            output[b * D2 + c] = sdata[0];
        }
    }
}

void sum_dim1_optimized(torch::Tensor input, torch::Tensor output) {
    const int B = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);
    
    // Use 256 threads per block for optimal occupancy
    const int threads_per_block = 256;
    const int blocks_per_grid_y = (D2 + threads_per_block - 1) / threads_per_block;
    
    dim3 block(threads_per_block);
    dim3 grid(B, blocks_per_grid_y);
    
    sum_dim1_final_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void sum_dim1_optimized(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1_optimized", &sum_dim1_optimized, "Optimized sum along dimension 1 with shared memory");
}
"""

# Compile the extension
sum_ext = load_inline(
    name='sum_dim1_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    """
    Reduce input tensor `x` of shape (B, D1, D2) along dimension 1 using a
    custom CUDA kernel optimized with shared memory and warp-level primitives.
    """
    assert dim == 1, "Only reduction along dimension 1 is supported."
    
    B, D1, D2 = x.shape
    
    # Output tensor with shape (B, D2)
    output = torch.zeros((B, D2), device=x.device, dtype=x.dtype)
    
    # Launch the optimized kernel
    sum_ext.sum_dim1_optimized(x, output)
    
    # Reshape to (B, 1, D2) to match expected output format
    return output.unsqueeze(1)

# Quick sanity-check when the file is executed directly
if __name__ == "__main__":
    batch_size = 128
    dim1 = 4096
    dim2 = 4095
    reduce_dim = 1

    x = torch.rand(batch_size, dim1, dim2, device='cuda', dtype=torch.float32)
    out = functional_model(x, dim=reduce_dim)

    # Shape check
    assert out.shape == (batch_size, 1, dim2), f"Wrong shape: {out.shape}"

    # Result check against PyTorch's native reduction
    ref = x.sum(dim=1, keepdim=True)
    assert torch.allclose(out, ref, atol=1e-4), "Results diverge!"
    print("✓ functional_model with optimized kernel works correctly.")
