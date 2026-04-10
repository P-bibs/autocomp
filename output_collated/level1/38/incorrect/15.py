# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_024918/code_28.py
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
# CUDA source – Single fused kernel:
# Performs row-wise reduction and element-wise scaling in one launch.
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_normalize_kernel(const float *x,
                                       float *output,
                                       const int batch_size,
                                       const int dim)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    // 1) Compute partial sum of |x| for the row in registers
    float sum = 0.0f;
    for (int col = tid; col < dim; col += stride) {
        sum += fabsf(x[row * dim + col]);
    }

    // 2) Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // 3) Use shared memory to make the total sum available to all threads in block
    __shared__ float row_sum;
    if (tid == 0) row_sum = sum;
    __syncthreads();

    // 4) Element-wise normalization
    // Reuse the already-read row_sum from shared memory
    float inv_sum = static_cast<float>(dim) / row_sum;
    for (int col = tid; col < dim; col += stride) {
        int idx = row * dim + col;
        output[idx] = x[idx] * inv_sum;
    }
}

void fused_normalize(torch::Tensor x, torch::Tensor output)
{
    const int batch_size = x.size(0);
    const int dim        = x.size(1);
    const int threads    = 256;
    
    // One block per row for massive parallelism efficiency
    fused_normalize_kernel<<<batch_size, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, dim);
}
"""

# ----------------------------------------------------------------------
# C++ binding
# ----------------------------------------------------------------------
cpp_source = r"""
void fused_normalize(torch::Tensor x, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize, "Fused abs-reduction + normalization");
}
"""

# ----------------------------------------------------------------------
# Compile the inline CUDA extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_norm',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize each row of x by the mean of its absolute values.
    Optimized via single-pass kernel fusion using warp primitives.
    """
    if not x.is_cuda:
        x = x.cuda()

    # Output buffer pre-allocation
    output = torch.empty_like(x)

    # Launch fused kernel
    fused_ext.fused_normalize(x, output)

    return output

# ----------------------------------------------------------------------
# Helper functions for the harness/testing
# ----------------------------------------------------------------------
def get_init_inputs():
    return []

def get_inputs():
    # Setup for benchmarking as specified
    batch_size = 32768
    dim = 65535
    x = torch.rand(batch_size, dim, dtype=torch.float32, device='cuda')
    return [x]
