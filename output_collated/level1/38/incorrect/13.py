# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_024918/code_12.py
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
# CUDA source – fused kernel to reduce kernel-launch overhead
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Fused kernel: one block per row, performs reduction and normalization
__global__ void fused_normalize_kernel(const float *x,
                                       float *output,
                                       const int batch_size,
                                       const int dim)
{
    const int row   = blockIdx.x;                // one block per row
    const int tid   = threadIdx.x;
    const int stride = blockDim.x;

    // ---- 1) compute row-wise sum of |x| ----
    float sum = 0.0f;
    for (int col = tid; col < dim; col += stride) {
        float v = x[row * dim + col];
        sum += fabsf(v);
    }

    // ---- warp-level reduction (no __syncthreads) ----
    // each warp (32 threads) reduces its partial sums
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // ---- store the final row sum in shared memory ----
    __shared__ float row_sum;
    if (tid == 0) row_sum = sum;
    __syncthreads();          // only once per block

    // ---- 2) element-wise scaling ----
    // each thread writes one output (or a strided loop if you want to process more)
    for (int col = tid; col < dim; col += stride) {
        int idx   = row * dim + col;
        float val = x[idx];
        // normalise: x * dim / sum_abs  (division by zero gives ±inf, same as original)
        float out = val * static_cast<float>(dim) / row_sum;
        output[idx] = out;
    }
}

// Host launcher for the fused kernel
void fused_normalize(torch::Tensor x, torch::Tensor /*sum_abs*/, torch::Tensor output)
{
    const int batch_size = x.size(0);
    const int dim        = x.size(1);
    const int threads    = 256;                 // multiple of 32, good occupancy
    const int blocks     = batch_size;           // one block per row

    fused_normalize_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, dim);
    cudaDeviceSynchronize();
}
"""

# ----------------------------------------------------------------------
# C++ binding – exposes fused_normalize to Python
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_normalize(torch::Tensor x,
                     torch::Tensor sum_abs,
                     torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize,
          "Fused abs-reduction + normalization kernel");
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

# ----------------------------------------------------------------------
# functional_model – the only function imported during evaluation
# ----------------------------------------------------------------------
def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize each row of x by the mean of its absolute values.
    Equivalent to: x / torch.mean(torch.abs(x), dim=1, keepdim=True)
    """
    # Ensure the input lives on the GPU
    if not x.is_cuda:
        x = x.cuda()

    batch_size = x.size(0)
    dim        = x.size(1)

    # Allocate temporary buffer for per-row sums and output
    sum_abs = torch.empty(batch_size, dtype=x.dtype, device=x.device)  # unused but kept for interface compatibility
    output  = torch.empty_like(x)

    # Launch fused kernel
    fused_ext.fused_normalize(x, sum_abs, output)

    return output


# ----------------------------------------------------------------------
# Helper functions required by the harness
# ----------------------------------------------------------------------
def get_init_inputs():
    return []

def get_inputs():
    batch_size = 32768
    dim = 65535
    # Random input matching the original benchmark shape/dtype
    x = torch.rand(batch_size, dim, dtype=torch.float32)
    return [x]
