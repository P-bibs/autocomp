# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_020747/code_12.py
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
# CUDA source – optimized with __restrict__ and removed sync points
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel 1: per‑row reduction – one block per row
// Using standard warp-level reduction patterns would be faster, 
// but we maintain the block-level logic for stability.
__global__ void reduce_sum_abs_kernel(const float * __restrict__ x,
                                      float * __restrict__ sum_abs,
                                      const int dim)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    float sum = 0.0f;
    for (int col = tid; col < dim; col += stride) {
        sum += fabsf(x[row * dim + col]);
    }

    __shared__ float sdata[256];
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sum_abs[row] = sdata[0];
    }
}

// Kernel 2: element‑wise scaling
__global__ void scale_kernel(const float * __restrict__ x,
                             const float * __restrict__ sum_abs,
                             float * __restrict__ output,
                             const int batch_size,
                             const int dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;
    if (idx >= total) return;

    const int row = idx / dim;
    // Normalized value x * dim / sum_abs
    output[idx] = x[idx] * (static_cast<float>(dim) / sum_abs[row]);
}

void fused_normalize(torch::Tensor x, torch::Tensor sum_abs, torch::Tensor output)
{
    const int batch_size = x.size(0);
    const int dim = x.size(1);
    const int threads = 256;

    // Launch reduction: 1 block per row
    reduce_sum_abs_kernel<<<batch_size, threads>>>(
        x.data_ptr<float>(), sum_abs.data_ptr<float>(), dim);

    // Launch scale: grid-based tiling
    const int total = batch_size * dim;
    const int blocks_s = (total + threads - 1) / threads;
    scale_kernel<<<blocks_s, threads>>>(
        x.data_ptr<float>(), sum_abs.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_normalize(torch::Tensor x, torch::Tensor sum_abs, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize, "Optimized fused normalization");
}
"""

# Compile extension
fused_ext = load_inline(
    name='fused_norm_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        x = x.cuda()
    
    batch_size, dim = x.shape
    sum_abs = torch.empty(batch_size, dtype=x.dtype, device=x.device)
    output = torch.empty_like(x)

    # Execution is non-blocking on the host; kernels queue in same stream
    fused_ext.fused_normalize(x, sum_abs, output)
    
    return output

def get_init_inputs():
    return []

def get_inputs():
    # Shape provided in requirements
    batch_size, dim = 32768, 65535
    x = torch.rand(batch_size, dim, dtype=torch.float32)
    return [x]
