# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_030939/code_2.py
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

# --- Optimized CUDA Kernel Source ---
cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void fused_normalize_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const int N,
    const int D
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    
    // Calculate grid dimensions
    int rows_per_block = min(32, (blockDim.x + D - 1) / D);
    int blocks_needed = (N + rows_per_block - 1) / rows_per_block;
    int block_row_start = blockIdx.x * rows_per_block;
    
    if (block_row_start >= N) return;
    
    int block_rows = min(rows_per_block, N - block_row_start);
    float* row_sums = sdata;
    
    // Phase 1: Compute sum of absolute values for each row in this block
    for (int i = 0; i < block_rows; i++) {
        int row_idx = block_row_start + i;
        float local_sum = 0.0f;
        
        // Each thread processes multiple elements in the row for better coalescing
        for (int d = tid; d < D; d += blockDim.x) {
            local_sum += fabsf(x[row_idx * D + d]);
        }
        
        // Warp-level reduction for this row's sum
        local_sum = warp_reduce_sum(local_sum);
        
        // Store warp results in shared memory
        if (lane_id == 0) {
            row_sums[i * ((blockDim.x + 31) / 32) + warp_id] = local_sum;
        }
    }
    __syncthreads();
    
    // Phase 2: Reduce across warps for each row
    for (int i = 0; i < block_rows; i++) {
        if (warp_id == 0) {
            float warp_sum = (lane_id < (blockDim.x + 31) / 32) ? 
                              row_sums[i * ((blockDim.x + 31) / 32) + lane_id] : 0.0f;
            warp_sum = warp_reduce_sum(warp_sum);
            
            if (lane_id == 0) {
                row_sums[i] = warp_sum;
            }
        }
    }
    __syncthreads();
    
    // Phase 3: Normalize each row
    for (int i = 0; i < block_rows; i++) {
        int row_idx = block_row_start + i;
        float mean_inv = (float)D / row_sums[i];
        
        // Coalesced write to output
        for (int d = tid; d < D; d += blockDim.x) {
            out[row_idx * D + d] = x[row_idx * D + d] * mean_inv;
        }
    }
}

void launch_fused_normalize(const at::Tensor& x, at::Tensor& out) {
    const int N = x.size(0);
    const int D = x.size(1);
    
    // Use fewer blocks with more rows per block for better utilization
    const int threads_per_block = 512;
    const int rows_per_block = min(32, (threads_per_block + D - 1) / D);  // Heuristic
    const int num_blocks = (N + rows_per_block - 1) / rows_per_block;
    
    // Shared memory: one float per warp per row + extra for alignment
    const int shared_mem_size = rows_per_block * ((threads_per_block + 31) / 32) * sizeof(float);
    
    fused_normalize_forward_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        N, 
        D
    );
}
'''

# --- C++ Interface ---
cpp_source = r'''
#include <torch/extension.h>
void launch_fused_normalize(const at::Tensor& x, at::Tensor& out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &launch_fused_normalize, "Fused Normalize Forward");
}
'''

# Compile JIT
fused_module = load_inline(
    name='fused_normalize',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    Optimized functional_model with improved memory coalescing:
    - Processes multiple rows per thread block
    - Ensures coalesced memory accesses
    - Reduces launch overhead
    """
    # Ensure contiguous buffer for coalesced memory access
    if not x.is_contiguous():
        x = x.contiguous()
        
    out = torch.empty_like(x)
    fused_module.forward(x, out)
    return out

# --- Compatibility requirements ---
batch_size = 32768
dim = 65535

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]
