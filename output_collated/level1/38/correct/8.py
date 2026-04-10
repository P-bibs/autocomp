# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_021806/code_8.py
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
# Optimized CUDA implementation
# The strategy:
# 1. Use 1 block per row to handle high-latency reduction entirely in shared memory.
# 2. Fuse the reduction and the normalization into a single pass to eliminate
#    intermediate global memory storage and kernel launch overhead.
# ----------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Fused kernel: performs reduction on a row, then immediately normalizes the row.
// Each block processes exactly one row.
__global__ void fused_normalize_kernel(const float* __restrict__ x,
                                       float* __restrict__ output,
                                       const int dim)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Shared memory for block-wide reduction
    __shared__ float sdata[256];
    
    float row_abs_sum = 0.0f;
    const float* row_ptr = x + row * dim;
    float* out_ptr = output + row * dim;

    // 1. Partial reduction
    for (int col = tid; col < dim; col += blockDim.x) {
        row_abs_sum += fabsf(row_ptr[col]);
    }

    sdata[tid] = row_abs_sum;
    __syncthreads();

    // 2. Parallel reduction within shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Single thread broadcasts divisor
    float inv_sum = 1.0f / (sdata[0] + 1e-20f); // Add tiny epsilon to avoid div by zero
    float factor = static_cast<float>(dim) * inv_sum;

    // 3. Normalization pass
    for (int col = tid; col < dim; col += blockDim.x) {
        out_ptr[col] = row_ptr[col] * factor;
    }
}

void fused_normalize_forward(torch::Tensor x, torch::Tensor output)
{
    const int batch_size = x.size(0);
    const int dim = x.size(1);
    
    // Launch one block per row
    fused_normalize_kernel<<<batch_size, 256>>>(
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
    m.def("fused_normalize", &fused_normalize_forward, "Fused L1-normalization kernel");
}
"""

# Compile the inline CUDA extension
fused_ext = load_inline(
    name='fused_norm',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized normalization: Performs the reduction and scaling in a single
    fused kernel launch, minimizing global memory round-trips and synchronization.
    """
    if not x.is_cuda:
        x = x.cuda()

    # Pre-allocate output only
    output = torch.empty_like(x)
    
    # Execute fused kernel
    fused_ext.fused_normalize(x, output)

    return output

def get_init_inputs():
    return []

def get_inputs():
    batch_size = 32768
    dim = 65535
    # Generate random input for testing
    x = torch.rand(batch_size, dim, dtype=torch.float32).cuda()
    return [x]
