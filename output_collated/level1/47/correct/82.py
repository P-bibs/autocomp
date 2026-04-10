# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_141159/code_21.py
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
#  Optimized CUDA Source: Grid-stride reduction with coalesced access
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_dim1_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                const int B, const int D1, const int D2) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = B * D2;

    // Grid-stride loop: ensures hardware utilization regardless of problem size
    for (int idx = tid; idx < total_elements; idx += blockDim.x * gridDim.x) {
        int b = idx / D2;
        int d2 = idx % D2;

        float sum = 0.0f;
        const float* __restrict__ data = input + b * D1 * D2 + d2;

        // Perform reduction in global memory by jumping D2 (coalesced within the column)
        // Accessing input[i * D2 + d2] is strided by D2
        #pragma unroll 4
        for (int i = 0; i < D1; ++i) {
            sum += data[i * D2];
        }
        output[idx] = sum;
    }
}

void sum_dim1_launcher(torch::Tensor input, torch::Tensor output) {
    const int B = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);
    const int total_elements = B * D2;

    // Heuristics for grid/block configuration
    const int threads = 256;
    const int blocks = std::min((total_elements + threads - 1) / threads, 2048);

    sum_dim1_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void sum_dim1_launcher(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1", &sum_dim1_launcher, "Optimized sum along dim 1");
}
"""

# Compile the extension
sum_ext = load_inline(
    name='sum_dim1',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    """
    Optimized reduction along dimension 1 using a grid-stride CUDA kernel.
    """
    assert dim == 1, "Only reduction along dimension 1 is supported."
    
    B, D1, D2 = x.shape
    # Ensure output is on the same device as input
    output = torch.empty((B, D2), device=x.device, dtype=x.dtype)
    
    # Launch kernel
    sum_ext.sum_dim1(x, output)
    
    return output.unsqueeze(1)

# -------------------------------------------------------------------------
#  Sanity check
# -------------------------------------------------------------------------
if __name__ == "__main__":
    batch_size, dim1, dim2 = 128, 4096, 4095
    x = torch.rand(batch_size, dim1, dim2, device='cuda', dtype=torch.float32)
    
    out = functional_model(x, dim=1)
    ref = x.sum(dim=1, keepdim=True)
    
    assert torch.allclose(out, ref, atol=1e-4), "Results diverge!"
    print("✓ functional_model works correctly.")
