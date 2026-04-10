# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_020747/code_14.py
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
# Optimized CUDA implementation:
# 1. Single kernel for reduction + normalization.
# 2. Uses shared memory for row-sum reduction.
# 3. Optimized access patterns with __restrict__ and __ldg.
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// We use 256 threads per block. Each block handles one row.
#define BLOCK_SIZE 256

__global__ void fused_normalize_kernel(const float* __restrict__ x,
                                       float* __restrict__ output,
                                       const int rows,
                                       const int cols)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Shared memory for reduction within the block
    __shared__ float sdata[BLOCK_SIZE];

    // Phase 1: Local partial reduction
    float partial_sum = 0.0f;
    for (int col = tid; col < cols; col += BLOCK_SIZE) {
        partial_sum += fabsf(__ldg(&x[row * cols + col]));
    }
    sdata[tid] = partial_sum;
    __syncthreads();

    // Phase 2: Parallel reduction within shared memory
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    const float row_sum = sdata[0];
    const float inv_sum = (row_sum != 0.0f) ? (static_cast<float>(cols) / row_sum) : (1.0f / 0.0f);

    // Phase 3: Normalize and write to global memory
    for (int col = tid; col < cols; col += BLOCK_SIZE) {
        int idx = row * cols + col;
        output[idx] = __ldg(&x[idx]) * inv_sum;
    }
}

void fused_normalize(torch::Tensor x, torch::Tensor output)
{
    const int rows = x.size(0);
    const int cols = x.size(1);
    
    // Launch one block per row
    fused_normalize_kernel<<<rows, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_normalize(torch::Tensor x, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize, "Fused abs reduction and normalization kernel");
}
"""

# Compile the extension with architecture-specific flags for 2080Ti (sm_75)
fused_ext = load_inline(
    name='fused_norm',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized normalization: x * dim / sum(abs(x), dim=1)
    Performed in a single fused GPU kernel launch.
    """
    if not x.is_cuda:
        x = x.cuda()

    output = torch.empty_like(x)
    fused_ext.fused_normalize(x, output)
    return output

def get_init_inputs():
    return []

def get_inputs():
    # Use the specified shape: 32768 x 65535
    batch_size = 32768
    dim = 65535
    x = torch.rand(batch_size, dim, dtype=torch.float32)
    return [x]
