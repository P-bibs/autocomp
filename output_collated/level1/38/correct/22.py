# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_030939/code_3.py
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

cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_normalize_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const int N,
    const int D
) {
    // Process multiple rows per block to reduce launch overhead
    const int rows_per_block = 4;
    const int actual_rows = min(rows_per_block, N - blockIdx.x * rows_per_block);
    
    // Shared memory for storing sums of each row
    extern __shared__ float sdata[];
    float* row_sums = sdata;
    
    int base_row = blockIdx.x * rows_per_block;
    
    // Each thread processes elements across multiple rows
    for (int row_idx = 0; row_idx < actual_rows; row_idx++) {
        int row = base_row + row_idx;
        float local_sum = 0.0f;
        
        // Phase 1: Compute L1 norm for this row
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            local_sum += fabsf(x[row * D + d]);
        }
        
        // Warp-level reduction for partial sums
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }
        
        // Store warp sums in shared memory
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;
        
        if (lane == 0) {
            row_sums[row_idx * (blockDim.x / 32) + wid] = local_sum;
        }
        __syncthreads();
        
        // Final reduction of warp sums by first warp
        if (wid == 0) {
            float warp_sum = (lane < (blockDim.x / 32)) ? 
                              row_sums[row_idx * (blockDim.x / 32) + lane] : 0.0f;
            
            for (int offset = blockDim.x / 64; offset > 0; offset /= 2) {
                warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
            }
            
            if (lane == 0) {
                row_sums[row_idx] = (float)D / warp_sum;
            }
        }
        __syncthreads();
    }
    
    // Phase 2: Normalize each row
    for (int row_idx = 0; row_idx < actual_rows; row_idx++) {
        int row = base_row + row_idx;
        float mean_inv = row_sums[row_idx];
        
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            out[row * D + d] = x[row * D + d] * mean_inv;
        }
    }
}

void launch_fused_normalize(const at::Tensor& x, at::Tensor& out) {
    const int N = x.size(0);
    const int D = x.size(1);
    
    // Process 4 rows per block
    const int rows_per_block = 4;
    const int threads = 256;  // Reduced threads per block for better occupancy
    const int blocks = (N + rows_per_block - 1) / rows_per_block;
    
    // Shared memory: 4 rows * (256/32) warps = 32 floats
    size_t smem = rows_per_block * (threads / 32) * sizeof(float);
    
    fused_normalize_forward_kernel<<<blocks, threads, smem>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), N, D
    );
}
'''

cpp_source = r'''
#include <torch/extension.h>
void launch_fused_normalize(const at::Tensor& x, at::Tensor& out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &launch_fused_normalize, "Fused Normalize Forward");
}
'''

fused_module = load_inline(
    name='fused_normalize',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(x):
    """
    Optimized normalization using cooperative processing of multiple rows per block
    to reduce kernel launch overhead and increase arithmetic intensity.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    fused_module.forward(x, out)
    return out
