# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_032736/code_18.py
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

# ----------------------------------------------------------------------
# Optimized CUDA source: 
# Single-pass fused reduction and normalization kernel.
# Uses shared memory and warp shuffle primitives for high-speed reduction.
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define THREADS_PER_BLOCK 256

// Optimized warp-level reduction using shuffle instructions
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void fused_normalize_kernel(
    const float *__restrict__ x,
    float *__restrict__ output,
    const int dim)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float *row_ptr = x + (row * dim);
    float *out_ptr = output + (row * dim);

    // Step 1: Compute partial sum of |x| in registers
    float local_sum = 0.0f;
    for (int col = tid; col < dim; col += THREADS_PER_BLOCK) {
        local_sum += fabsf(row_ptr[col]);
    }

    // Step 2: Shared memory reduction
    __shared__ float sdata[THREADS_PER_BLOCK];
    sdata[tid] = local_sum;
    __syncthreads();

    // Reduce shared memory
    for (int s = THREADS_PER_BLOCK / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Final warp-level reduction
    float final_sum = (tid < 32) ? warp_reduce_sum(sdata[tid]) : 0.0f;
    
    // Broadcast the sum back to all threads in the block
    if (tid == 0) sdata[0] = final_sum;
    __syncthreads();
    float total_abs_sum = sdata[0];

    // Step 3: Scaling phase (re-reads input once, writes output once)
    const float scale_factor = (total_abs_sum == 0.0f) ? 0.0f : (static_cast<float>(dim) / total_abs_sum);
    for (int col = tid; col < dim; col += THREADS_PER_BLOCK) {
        out_ptr[col] = row_ptr[col] * scale_factor;
    }
}

void fused_normalize_forward(torch::Tensor x, torch::Tensor output) {
    const int batch_size = x.size(0);
    const int dim = x.size(1);
    
    // Launch one block per row
    fused_normalize_kernel<<<batch_size, THREADS_PER_BLOCK>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        dim
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_normalize_forward(torch::Tensor x, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize_forward, "Fused normalization kernel");
}
"""

fused_ext = load_inline(
    name='fused_norm_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized normalization: x / (sum(|x|, dim=1) / dim)
    Uses a single fused CUDA kernel for minimal memory throughput.
    """
    if not x.is_cuda:
        x = x.cuda()
    
    output = torch.empty_like(x)
    fused_ext.fused_normalize(x, output)
    return output

def get_init_inputs():
    return []

def get_inputs():
    batch_size = 32768
    dim = 65535
    x = torch.rand(batch_size, dim, dtype=torch.float32)
    return [x]
