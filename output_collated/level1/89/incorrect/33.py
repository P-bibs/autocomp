# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_081907/code_5.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['dim']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    A simple model that performs a cumulative sum (prefix sum) operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the scan operation.
    """

    def __init__(self, dim):
        """
        Initialize the Scan model.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
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

# Optimization: Using Warp-level Primitives for parallel prefix scan
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define MAX_WARPS_PER_BLOCK 32

__device__ inline float warp_scan(float val) {
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float temp = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (threadIdx.x % WARP_SIZE >= offset)
            val += temp;
    }
    return val;
}

__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void scan_kernel(const float* input, float* output, int batch_size, int seq_len) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* row_input = input + batch_idx * seq_len;
    float* row_output = output + batch_idx * seq_len;
    
    // Method: Use multiple warps per row with hierarchical scan
    // Phase 1: In-warp scans
    int warps_needed = (seq_len + WARP_SIZE - 1) / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    // Shared memory to store warp sums
    extern __shared__ float warp_sums[];
    float thread_val = 0;
    
    // Each warp performs its partial scan
    if (warp_id < warps_needed) {
        int start_idx = warp_id * WARP_SIZE;
        int idx = start_idx + lane_id;
        
        if (idx < seq_len) {
            thread_val = row_input[idx];
        }
        
        // Warp-level inclusive scan
        thread_val = warp_scan(thread_val);
        
        // Store the warp sum (last element of the warp's scan)
        if (lane_id == WARP_SIZE - 1) {
            warp_sums[warp_id] = thread_val;
        }
    } else {
        warp_sums[warp_id] = 0.0f;
    }
    
    __syncthreads();
    
    // Phase 2: Scan the warp sums (only first few threads participate)
    if (threadIdx.x < warps_needed) {
        float val = warp_sums[threadIdx.x];
        val = warp_scan(val);
        warp_sums[threadIdx.x] = val;
    }
    __syncthreads();
    
    // Phase 3: Add warp prefix to each thread's value
    if (warp_id < warps_needed) {
        float warp_prefix = (warp_id > 0) ? warp_sums[warp_id - 1] : 0.0f;
        int start_idx = warp_id * WARP_SIZE;
        int idx = start_idx + lane_id;
        
        if (idx < seq_len) {
            row_output[idx] = thread_val + warp_prefix;
        }
    }
}

void cumsum_cuda(torch::Tensor input, torch::Tensor output) {
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    
    // Calculate shared memory size needed
    int warps_needed = (seq_len + WARP_SIZE - 1) / WARP_SIZE;
    int shared_mem_bytes = warps_needed * sizeof(float);
    
    dim3 blocks(batch_size);
    // Use enough threads to cover all warps needed, up to hardware limit
    dim3 threads(min(warps_needed * WARP_SIZE, MAX_WARPS_PER_BLOCK * WARP_SIZE));
    
    scan_kernel<<<blocks, threads, shared_mem_bytes>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        seq_len
    );
}
"""

cpp_source = r"""
void cumsum_cuda(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cumsum", &cumsum_cuda, "High-performance cumsum using CUDA warp primitives");
}
"""

fused_ext = load_inline(
    name='cumsum_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    # Only supports dim=1 based on problem description
    output = torch.empty_like(x)
    fused_ext.cumsum(x.contiguous(), output)
    return output

# Verification/Usage
batch_size = 32768
input_shape = (32768,)
dim = 1

def get_init_inputs():
    return [dim]

def get_inputs():
    return [torch.rand(batch_size, *input_shape)]
