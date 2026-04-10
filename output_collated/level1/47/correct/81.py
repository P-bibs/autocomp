# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_141159/code_19.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tile-based reduction to ensure global memory coalescing
__global__ void sum_dim1_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                int B, int D1, int D2) {
    int b = blockIdx.x;
    int tx = threadIdx.x;
    
    // Each thread block processes a chunk of D2 elements
    for (int col = blockIdx.y * blockDim.x + tx; col < D2; col += gridDim.y * blockDim.x) {
        float sum = 0.0f;
        const float* input_ptr = input + b * D1 * D2 + col;
        
        // Summing along D1 dimension
        #pragma unroll 4
        for (int i = 0; i < D1; ++i) {
            sum += input_ptr[i * D2];
        }
        
        output[b * D2 + col] = sum;
    }
}

void sum_dim1(torch::Tensor input, torch::Tensor output) {
    const int B = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);
    
    // Threads: 256 is generally optimal for memory-bound reductions
    const int threads = 256;
    // We want enough blocks to occupy the GPU, but ensure each block 
    // handles a decent chunk of D2 to keep memory requests coalesced
    dim3 grid(B, (D2 + threads - 1) / threads);
    if (grid.y > 65535) grid.y = 65535; // Cap to hardware limits
    
    sum_dim1_kernel<<<grid, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), B, D1, D2);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void sum_dim1(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1", &sum_dim1, "Optimized sum along dim 1");
}
"""

sum_ext = load_inline(
    name='sum_dim1_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    assert dim == 1
    # Create output: (B, D2)
    output = torch.empty((x.shape[0], x.shape[2]), device=x.device, dtype=x.dtype)
    sum_ext.sum_dim1(x, output)
    # Return as (B, 1, D2)
    return output.unsqueeze(1)
