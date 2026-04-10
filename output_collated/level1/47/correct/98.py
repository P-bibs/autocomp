# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_143434/code_31.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['dim']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs sum reduction over a specified dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
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

# -------------------------------------------------------------------------
# CUDA kernel source – reduction using warp‑level primitives
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 256;

__global__ void sum_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch,
    const int dim1,
    const int dim2)
{
    int idx = blockIdx.x;
    if (idx >= batch * dim2) return;
    int b = idx / dim2;
    int d2 = idx % dim2;

    const float* slice = input + (b * dim1 * dim2 + d2);
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    float sum = 0.0f;
    
    // Coalesced access: threads read elements spaced by dim2 which 
    // are contiguous if we view the input as (batch, dim2, dim1)
    // However, original input is (batch, dim1, dim2). 
    // We adjust indexing to maintain coalescing.
    for (int i = tid; i < dim1; i += BLOCK_SIZE) {
        sum += input[b * dim1 * dim2 + i * dim2 + d2];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        float val = sdata[tid];
        if (BLOCK_SIZE >= 64) val += sdata[tid + 32];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (tid == 0) output[idx] = val;
    }
}

void fused_op_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    int batch,
    int dim1,
    int dim2)
{
    const int blocks = batch * dim2;
    sum_reduce_kernel<<<blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, dim1, dim2);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(const torch::Tensor& input, torch::Tensor& output, int batch, int dim1, int dim2);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Sum reduction kernel");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    if dim != 1:
        return torch.sum(x, dim=dim, keepdim=True)
        
    batch, dim1, dim2 = x.shape
    # Ensure contiguous memory for correct indexing in custom kernel
    x = x.contiguous()
    output = torch.empty((batch, 1, dim2), dtype=x.dtype, device=x.device)
    
    fused_ext.fused_op(x, output, batch, dim1, dim2)
    return output
