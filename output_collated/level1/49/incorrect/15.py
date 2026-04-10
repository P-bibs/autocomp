# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_152339/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['dim']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs Max reduction over a specific dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
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
# CUDA kernel: Performs a max reduction along the innermost contiguous dim
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void max_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N,
    const int red_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float* row = input + (size_t)idx * red_size;
    float max_val = -3.402823466e+38F; // -FLT_MAX

    // Unrolling allows the compiler to pipeline memory loads
    #pragma unroll 8
    for (int i = 0; i < red_size; ++i) {
        float val = row[i];
        if (val > max_val) max_val = val;
    }
    output[idx] = max_val;
}

void max_reduce_cuda(torch::Tensor input, torch::Tensor output) {
    int N = 1;
    for(int i = 0; i < input.dim() - 1; ++i) N *= input.size(i);
    int red_size = (int)input.size(input.dim() - 1);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    max_reduce_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        red_size
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_reduce_cuda(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reduce", &max_reduce_cuda, "Max reduction along last dim");
}
"""

# Compile the inline extension
max_ext = load_inline(
    name='max_reduce',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor, *, dim: int) -> torch.Tensor:
    """
    Optimized reduction using a custom CUDA kernel.
    """
    # 1. Bring reduction dim to end and flatten leading dims effectively
    x_moved = x.moveaxis(dim, -1)
    
    # 2. Flatten all leading dimensions to 1D
    original_shape = list(x.shape)
    reduced_shape = original_shape[:dim] + original_shape[dim+1:]
    
    # 3. Ensure contiguous memory for the kernel
    x_contig = x_moved.contiguous()
    
    # 4. Prepare output buffer
    N = 1
    for s in reduced_shape: N *= s
    out = torch.empty(N, dtype=x.dtype, device=x.device)
    
    # 5. Launch custom kernel
    max_ext.max_reduce(x_contig, out)
    
    return out.reshape(reduced_shape)

# Helper functions for the evaluation harness
batch_size, dim1, dim2 = 128, 4096, 4095

def get_init_inputs():
    return [1]

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device='cuda', dtype=torch.float32)
    return [x]
