# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_021806/code_0.py
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
# CUDA source – fused kernel:
#   1. reduce_sum_abs within block
#   2. normalize all elements in the row using shared result
# ----------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Fused kernel: per-row reduction + normalization
__global__ void fused_normalize_kernel(const float* x,
                                       float* output,
                                       const int batch_size,
                                       const int dim)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    // Shared memory for block reduction
    __shared__ float sdata[256]; // assumes <= 256 threads per block
    float sum = 0.0f;

    // Each thread sums every stride-th element
    for (int col = tid; col < dim; col += stride) {
        sum += fabsf(x[row * dim + col]);
    }

    sdata[tid] = sum;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Broadcast final sum to all threads
    float row_sum = sdata[0];
    __syncthreads(); // Ensure all threads see the final sum

    // Scale elements in this row
    for (int col = tid; col < dim; col += stride) {
        float val = x[row * dim + col];
        output[row * dim + col] = val * static_cast<float>(dim) / row_sum;
    }
}

// Host entry point
void fused_normalize_forward(torch::Tensor x, torch::Tensor output)
{
    const int batch_size = x.size(0);
    const int dim = x.size(1);
    const int threads = 256;
    const int blocks = batch_size;

    fused_normalize_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );
}
"""

# ----------------------------------------------------------------------
# C++ binding for fused function
# ----------------------------------------------------------------------

cpp_source = r"""
#include <torch/extension.h>

void fused_normalize_forward(torch::Tensor x, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize_forward,
          "Fused normalize kernel: per-row L1 normalization");
}
"""

# ----------------------------------------------------------------------
# Compile inline CUDA extension
# ----------------------------------------------------------------------

fused_ext = load_inline(
    name='fused_norm',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# functional_model – will be imported and timed
# ----------------------------------------------------------------------

def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize each row of x by its mean of absolute values.
    Equivalent to: x / torch.mean(torch.abs(x), dim=1, keepdim=True)
    """
    if not x.is_cuda:
        x = x.cuda()

    batch_size = x.size(0)
    dim        = x.size(1)

    output = torch.empty_like(x)
    fused_ext.fused_normalize(x, output)

    return output


# ----------------------------------------------------------------------
# Helper functions required by the harness
# ----------------------------------------------------------------------

def get_init_inputs():
    return []

def get_inputs():
    batch_size = 32768
    dim = 65535
    x = torch.rand(batch_size, dim, dtype=torch.float32)
    return [x]
