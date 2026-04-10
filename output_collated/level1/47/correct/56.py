# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_131936/code_28.py
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

# ------------------------------------------------------------------------------
# Optimized CUDA kernel:
# 1. Removed shared memory entirely (not needed for simple reduction).
# 2. Removed __syncthreads() (no data dependency between threads).
# 3. Accumulate directly into registers to maximize throughput.
# 4. Global memory reads are coalesced by assigning each thread consecutive indices.
# ------------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32 
#define ELEMENTS_PER_THREAD 4

__global__ void sum_dim1_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int B, int D1, int D2) {
    
    // Grid maps: each block handles a subset of the (B, D2) output space
    int b = blockIdx.z; 
    int j_base = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;
    
    if (b >= B || j_base >= D2) return;

    // Use registers for local accumulation
    float local_sum[ELEMENTS_PER_THREAD] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Reduction loop: Iterate through the reduction dimension (D1)
    // This access pattern reads input[b, i, j] which is highly coalesced
    for (int i = 0; i < D1; ++i) {
        int row_offset = (b * D1 + i) * D2;
        #pragma unroll
        for (int v = 0; v < ELEMENTS_PER_THREAD; ++v) {
            int j = j_base + v;
            if (j < D2) {
                local_sum[v] += input[row_offset + j];
            }
        }
    }

    // Write back result
    int out_row_offset = b * D2;
    #pragma unroll
    for (int v = 0; v < ELEMENTS_PER_THREAD; ++v) {
        int j = j_base + v;
        if (j < D2) {
            output[out_row_offset + j] = local_sum[v];
        }
    }
}

void sum_dim1(torch::Tensor input, torch::Tensor output) {
    int B = input.size(0);
    int D1 = input.size(1);
    int D2 = input.size(2);

    // Threads per block
    const int threads = 256;
    // Calculate grid dimensions
    // x: blocks along D2 dim, z: batch dimension
    int num_cols = (D2 + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
    dim3 block_dim(threads);
    dim3 grid_dim((num_cols + threads - 1) / threads, 1, B);

    sum_dim1_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        B, D1, D2
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void sum_dim1(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1", &sum_dim1, "Sum along dimension 1 optimized");
}
"""

# Compile extension
sum_ext = load_inline(
    name='sum_dim1',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    assert dim == 1
    # Check shape: (B, D1, D2)
    B, D1, D2 = x.shape
    output = torch.zeros((B, D2), device=x.device, dtype=x.dtype)
    
    # Launch kernel
    sum_ext.sum_dim1(x, output)
    
    return output.unsqueeze(1)
