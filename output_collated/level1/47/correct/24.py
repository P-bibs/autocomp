# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_125421/code_2.py
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
#  High-performance CUDA implementation with advanced ILP and reduction
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_dim1_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int B, int D1, int D2) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;

    if (b < B && j < D2) {
        const float* __restrict__ b_ptr = input + (b * D1 * D2) + j;
        
        // Use 8 accumulators for better ILP
        float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;
        float sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f, sum8 = 0.0f;
        int i = 0;

        // Unroll 8-way to maximize instruction-level parallelism
        for (; i < D1 - 7; i += 8) {
            sum1 += b_ptr[i * D2];
            sum2 += b_ptr[(i + 1) * D2];
            sum3 += b_ptr[(i + 2) * D2];
            sum4 += b_ptr[(i + 3) * D2];
            sum5 += b_ptr[(i + 4) * D2];
            sum6 += b_ptr[(i + 5) * D2];
            sum7 += b_ptr[(i + 6) * D2];
            sum8 += b_ptr[(i + 7) * D2];
        }

        // Handle remaining elements with 4-way unrolling if enough elements left
        for (; i < D1 - 3; i += 4) {
            sum1 += b_ptr[i * D2];
            sum2 += b_ptr[(i + 1) * D2];
            sum3 += b_ptr[(i + 2) * D2];
            sum4 += b_ptr[(i + 3) * D2];
        }

        // Handle final remaining elements
        for (; i < D1; ++i) {
            sum1 += b_ptr[i * D2];
        }

        // Final reduction of all partial sums
        float final_sum = (sum1 + sum2) + (sum3 + sum4);
        final_sum += (sum5 + sum6) + (sum7 + sum8);
        output[b * D2 + j] = final_sum;
    }
}

void sum_dim1_gpu(torch::Tensor input, torch::Tensor output) {
    int B = input.size(0);
    int D1 = input.size(1);
    int D2 = input.size(2);
    
    const int threads = 256;
    dim3 block(threads);
    dim3 grid((D2 + threads - 1) / threads, B);
    
    sum_dim1_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void sum_dim1_gpu(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1_gpu", &sum_dim1_gpu, "Highly optimized sum along dim 1 with 8-way ILP");
}
"""

# Compile the optimized extension
sum_ext = load_inline(
    name='sum_dim1_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    """
    Sum along dimension 1 using custom optimized CUDA kernel.
    Shape: (B, D1, D2) -> (B, 1, D2)
    """
    assert dim == 1, "Only dim=1 is supported."
    
    batch, d1, d2 = x.shape
    output = torch.zeros((batch, d2), device=x.device, dtype=x.dtype)
    
    # Kernel computes (B, D2) result
    sum_ext.sum_dim1_gpu(x, output)
    
    return output.view(batch, 1, d2)

if __name__ == "__main__":
    # Sanity check
    batch_size, d1, d2 = 32, 128, 64
    x = torch.randn(batch_size, d1, d2, device='cuda')
    expected = x.sum(dim=1, keepdim=True)
    actual = functional_model(x, dim=1)
    
    assert torch.allclose(actual, expected, atol=1e-4)
    print("Optimization successful: output matches torch.sum.")
