# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_032736/code_29.py
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

# The fused kernel strategy:
# Each block processes one row. This allows us to compute the reduction (sum of absolute values)
# and use shared memory (or registers) to hold that value locally. 
# We then use a second pass over the same row to perform the division,
# ensuring we read 'x' from global memory into registers once per phase.
# To handle large dimensions, we split the kernel logic into specialized phases.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void fused_normalize_kernel(const float * __restrict__ x,
                                       float * __restrict__ output,
                                       const int dim)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    
    // Phase 1: Row-wise sum of absolute values
    float sum = 0.0f;
    for (int col = tid; col < dim; col += stride) {
        sum += fabsf(x[row * dim + col]);
    }

    // Warp-level reduction
    sum = warp_reduce_sum(sum);

    // Shared memory for block-level reduction
    __shared__ float s_warp_sums[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) s_warp_sums[warp_id] = sum;
    __syncthreads();

    // Final reduction across warps
    if (warp_id == 0) {
        sum = (tid < (blockDim.x + 31) / 32) ? s_warp_sums[tid] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (tid == 0) s_warp_sums[0] = sum;
    }
    __syncthreads();

    const float sum_abs = s_warp_sums[0];
    const float inv_sum = 1.0f / (sum_abs + 1e-9f); // Add small epsilon for stability
    const float scale = static_cast<float>(dim) * inv_sum;

    // Phase 2: Normalization
    for (int col = tid; col < dim; col += stride) {
        output[row * dim + col] = x[row * dim + col] * scale;
    }
}

void fused_normalize(torch::Tensor x, torch::Tensor output)
{
    const int batch_size = x.size(0);
    const int dim = x.size(1);
    
    // 1024 threads provides good occupancy for large dim
    const int threads = 1024;
    const int blocks = batch_size;

    fused_normalize_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), dim);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_normalize(torch::Tensor x, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize, "Fused abs-sum normalization kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_norm_optimized_v2',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Normalizes each row of x: output[i, j] = x[i, j] * dim / sum(|x[i, :]|)
    """
    if not x.is_cuda:
        x = x.cuda()
    
    output = torch.empty_like(x)
    fused_ext.fused_normalize(x, output)
    return output

def get_init_inputs():
    return []

def get_inputs():
    # Large dimensions as specified
    batch_size = 32768
    dim = 65535
    x = torch.rand(batch_size, dim, dtype=torch.float32)
    return [x]
