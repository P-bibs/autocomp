# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_141159/code_24.py
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

# ------------------------------------------------------------------
#  CUDA kernel – grid‑stride version
#  Each thread handles one (batch, col) pair and reduces over dim 1.
#  This leverages the cache hierarchy efficiently for read-only access (input)
#  and minimizes latency by removing __syncthreads() and shared memory.
# ------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" __global__
void sum_dim1_kernel(const float* __restrict__ input,
                     float* __restrict__ output,
                     const int B,
                     const int D1,
                     const int D2)
{
    // Map thread indexing to a 2D grid logic for (Batch, Column)
    // blockIdx.x handles Batch index
    // blockIdx.y * blockDim.x + threadIdx.x handles Column index
    const int batch = blockIdx.x;
    
    // Grid-stride loop logic for columns
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    const int stride = gridDim.y * blockDim.x;

    if (batch >= B) return;

    while (col < D2) {
        float sum = 0.0f;
        
        // Loop over the reduction dimension (D1)
        // Global access pattern: 
        // input[batch * D1 * D2 + i * D2 + col]
        // Since i is inner loop, we are jumping by D2 columns.
        // For performance on large D1, this is efficient as it maximizes
        // throughput for a single (b, col) position.
        const int batch_offset = batch * D1 * D2;
        for (int i = 0; i < D1; ++i) {
            sum += input[batch_offset + i * D2 + col];
        }

        output[batch * D2 + col] = sum;

        col += stride;
    }
}

void sum_dim1(torch::Tensor input, torch::Tensor output)
{
    const int B  = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);

    int threads = 256;
    // We restrict y-dimension to prevent overflowing block index limits
    // but keep enough blocks to saturate the GPU.
    int blocks_y = (D2 + threads - 1) / threads;
    
    // Clamp blocks_y to GPU hardware limits if necessary, though 256*65535 is fine for 4095
    dim3 gridDim(B, blocks_y);
    dim3 blockDim(threads);

    sum_dim1_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void sum_dim1(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1", &sum_dim1, "Sum along dimension 1 using grid-stride kernels");
}
"""

# Compile the extension
sum_ext = load_inline(
    name='sum_dim1_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    """
    Optimized sum along dimension 1.
    """
    assert dim == 1
    # Create output storage: (B, D2)
    output = torch.empty((x.shape[0], x.shape[2]), device=x.device, dtype=x.dtype)
    
    # Launch kernel
    sum_ext.sum_dim1(x, output)
    
    # Return with original dimension structure (B, 1, D2)
    return output.unsqueeze(1)

# --- Benchmark/Harness requirements ---
batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def get_init_inputs():
    return [reduce_dim]

def get_inputs():
    return [torch.rand(batch_size, dim1, dim2, device='cuda')]
