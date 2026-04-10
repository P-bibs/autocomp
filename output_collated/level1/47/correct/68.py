# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_134650/code_21.py
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

// Optimized kernel using tiled memory access for coalescing
// Each block handles a tile of columns to maximize cache hit rates
__global__ void sum_dim1_optimized_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          const int B, const int D1, const int D2) {
    const int tid = threadIdx.x;
    const int b = blockIdx.y;
    const int d2_start = blockIdx.x * blockDim.x;
    
    if (b >= B) return;

    for (int d2 = d2_start + tid; d2 < D2; d2 += blockDim.x) {
        float sum = 0.0f;
        const float* col_ptr = input + b * D1 * D2 + d2;
        
        // Accumulate along the reduction dimension (D1)
        #pragma unroll
        for (int d1 = 0; d1 < D1; ++d1) {
            sum += col_ptr[d1 * D2];
        }
        output[b * D2 + d2] = sum;
    }
}

void sum_dim1_optimized(torch::Tensor input, torch::Tensor output) {
    const int B = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);

    // Threads per block usually 256 for balanced occupancy/register pressure
    const int threads = 256;
    const int blocks_x = (D2 + threads - 1) / threads;
    const int blocks_y = B;
    
    dim3 grid(blocks_x, blocks_y);
    
    sum_dim1_optimized_kernel<<<grid, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void sum_dim1_optimized(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1", &sum_dim1_optimized, "Optimized reduction along dim 1");
}
"""

# Compile the extension
sum_ext = load_inline(
    name='sum_dim1_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    """
    Reduced input tensor x of shape (B, D1, D2) along dimension 1.
    Uses tiling to ensure memory coalescing on current NVIDIA hardware.
    """
    assert dim == 1, "Only reduction along dimension 1 is supported."
    B, D1, D2 = x.shape
    output = torch.empty((B, D2), device=x.device, dtype=x.dtype)
    
    sum_ext.sum_dim1(x, output)
    
    return output.unsqueeze(1)

if __name__ == "__main__":
    # Standard check
    batch_size, dim1, dim2 = 128, 4096, 4095
    x = torch.rand(batch_size, dim1, dim2, device='cuda', dtype=torch.float32)
    out = functional_model(x, dim=1)
    
    ref = x.sum(dim=1, keepdim=True)
    assert torch.allclose(out, ref, atol=1e-4), "Results diverge!"
    print("✓ Optimized reduction complete.")
