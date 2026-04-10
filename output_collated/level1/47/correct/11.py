# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_122232/code_15.py
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
# Optimized CUDA implementation:
# 1. Loop-unrolling with #pragma unroll 16 to improve ILP.
# 2. Hoisted base pointer to minimize arithmetic in inner loop.
# 3. Memory Access: Threads now process contiguous indices in the D2
#    dimension, ensuring coalesced global memory loads.
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_dim1_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                const int B,
                                const int D1,
                                const int D2) {
    const int b = blockIdx.x;
    const int j = blockIdx.y * blockDim.x + threadIdx.x;

    if (b < B && j < D2) {
        // Point to the start of the current batch slice
        const float* __restrict__ in = input + b * ((size_t)D1 * D2);
        
        float sum = 0.0f;

        // Optimization: Unrolling allows the compiler to schedule multiple 
        // independent loads and additions, effectively hiding memory latency.
        #pragma unroll 16
        for (int i = 0; i < D1; ++i) {
            // Access is coalesced because 'j' is the fastest varying dimension
            sum += in[i * D2 + j];
        }

        output[b * D2 + j] = sum;
    }
}

void sum_dim1(torch::Tensor input, torch::Tensor output) {
    const int B  = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);

    // Optimized threading: 256 threads per block is usually sweet spot for RTX 2080 Ti
    const int threads = 256;
    dim3 block_dim(threads);
    dim3 grid_dim(B, (D2 + threads - 1) / threads);

    sum_dim1_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void sum_dim1(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1", &sum_dim1, "Sum along dimension 1 (optimized)");
}
"""

# Compile the extension
sum_ext = load_inline(
    name='sum_dim1_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

def functional_model(x, *, dim):
    """
    Sums tensor `x` over dimension 1 (B, D1, D2) -> (B, 1, D2).
    Uses custom CUDA kernel for performance.
    """
    if dim != 1:
        raise ValueError("Only dimension 1 reduction is supported.")
    
    # Pre-allocate output: (B, 1, D2)
    shape = list(x.shape)
    shape[1] = 1
    output = torch.zeros(shape, device=x.device, dtype=x.dtype)
    
    # Launch kernel
    sum_ext.sum_dim1(x, output)
    return output

# Evaluation constants
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_init_inputs():
    return [1]

def get_inputs():
    return [torch.rand(batch_size, dim1, dim2, device='cuda')]

if __name__ == "__main__":
    # Smoke test
    x = torch.rand(2, 4, 5, device='cuda')
    y = functional_model(x, dim=1)
    expected = x.sum(dim=1, keepdim=True)
    assert torch.allclose(y, expected, atol=1e-5)
    print("Correctness check passed.")
