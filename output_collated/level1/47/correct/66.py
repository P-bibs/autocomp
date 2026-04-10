# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_134650/code_19.py
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
# Optimized CUDA Kernel
# The reduction along dimension 1 (dim1=4096) is problematic for simple 
# parallelization if we don't utilize shared memory or L2 caching effectively.
# Since dim1 is large, we can parallelize over (batch, dim2) and use 
# thread-local accumulation. Given the memory layout (B, D1, D2), 
# reading across D1 (strided) is unavoidable. To improve performance, we 
# ensure the inner loop accesses memory that targets shared L1/L2 cache lines effectively.
# -----------------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_dim1_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                int B, int D1, int D2) {
    // Each thread is responsible for one output element (b, j)
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;

    if (b < B && j < D2) {
        float sum = 0.0f;
        const float* __restrict__ col_ptr = input + (b * D1 * D2) + j;
        
        // Unroll loop for better instruction-level parallelism
        #pragma unroll 8
        for (int i = 0; i < D1; ++i) {
            sum += col_ptr[i * D2];
        }
        
        output[b * D2 + j] = sum;
    }
}

void sum_dim1(torch::Tensor input, torch::Tensor output) {
    const int B = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);
    
    // Threads per block: choose 256 for a good balance of occupancy
    const int threads = 256;
    const dim3 block_dim(threads);
    const dim3 grid_dim((D2 + threads - 1) / threads, B);
    
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
    m.def("sum_dim1", &sum_dim1, "Sum along dimension 1");
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
    Optimized sum reduction along dimension 1.
    Input shape: (B, D1, D2)
    Output shape: (B, 1, D2)
    """
    assert dim == 1
    B, D1, D2 = x.shape
    output = torch.empty((B, D2), device=x.device, dtype=x.dtype)
    sum_ext.sum_dim1(x, output)
    return output.unsqueeze(1)

# --- Parameters ---
batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def get_init_inputs():
    return [reduce_dim]

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device='cuda')
    return [x]
