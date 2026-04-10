# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_131936/code_20.py
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

# -----------------------------------------------------------------------------
# Optimized CUDA Kernel:
# 1. We map each thread to a specific column index j.
# 2. Each thread iterates through the reduction dimension D1.
# 3. Memory access is coalesced because thread i reads data at offset i * D2.
# -----------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_dim1_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                int B, int D1, int D2) {
    int b = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (b >= B || j >= D2) return;
    
    float local_sum = 0.0f;
    const float* __restrict__ batch_ptr = input + (size_t)b * D1 * D2 + j;
    
    // Each thread accumulates column j across all rows i
    // The stride between rows is D2, which is optimal for coalescing 
    // when threads within a warp read consecutive 'j' values.
    #pragma unroll 4
    for (int i = 0; i < D1; ++i) {
        local_sum += batch_ptr[(size_t)i * D2];
    }
    
    output[(size_t)b * D2 + j] = local_sum;
}

void sum_dim1(torch::Tensor input, torch::Tensor output) {
    const int B = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);
    
    // 256 threads per block is usually a sweet spot for RTX GPUs
    const int threads = 256;
    const dim3 block_dim(threads);
    const dim3 grid_dim(B, (D2 + threads - 1) / threads);
    
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
    m.def("sum_dim1", &sum_dim1, "Optimized sum along dim 1");
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
    Optimized sum along dim 1.
    Input shape: (B, D1, D2)
    Output shape: (B, 1, D2)
    """
    assert dim == 1, "Only dimension 1 reduction is supported"
    
    batch_size, _, dim2 = x.shape
    # Setup output tensor
    output = torch.empty((batch_size, dim2), device=x.device, dtype=x.dtype)
    
    # Call compiled CUDA kernel
    sum_ext.sum_dim1(x, output)
    
    # Unsqueeze to match expectation of (B, 1, D2)
    return output.unsqueeze(1)
