# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_152959/code_5.py
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

# CUDA Kernel Implementation
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void max_dim1_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                 int B, int D1, int D2) {
    // Map each thread to one column (d2) in one batch index (b)
    // We process the reduction across D1 (dim 1)
    int b = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;

    if (col < D2) {
        float max_val = -FLT_MAX;
        // Pointer to the start of the (batch, col) strip
        // Input layout is [B, D1, D2]. The strip is indexed by d1 * D2 + col.
        const float* input_ptr = input + b * (D1 * D2) + col;
        
        for (int d1 = 0; d1 < D1; ++d1) {
            float val = input_ptr[d1 * D2];
            if (val > max_val) max_val = val;
        }
        output[b * D2 + col] = max_val;
    }
}

void max_dim1(torch::Tensor input, torch::Tensor output) {
    const auto sizes = input.sizes();
    const int B = sizes[0];
    const int D1 = sizes[1];
    const int D2 = sizes[2];
    
    // Threads per block
    const int threads = 256;
    // Blocks: B for batch, ceil(D2/threads) for width
    dim3 grid(B, (D2 + threads - 1) / threads);
    
    max_dim1_kernel<<<grid, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), B, D1, D2
    );
}
"""

# C++ Wrapper for Pybind11
cpp_source = r"""
#include <torch/extension.h>

void max_dim1(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_dim1", &max_dim1, "Custom max reduction along dim 1");
}
"""

# Compile the inline extension
fused_ext = load_inline(
    name='max_dim1_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    """
    Optimized functional model using a custom CUDA kernel for max reduction.
    Assumes dim == 1 and x is float32 on CUDA.
    """
    if dim != 1:
        raise ValueError("This optimized kernel only supports reduction along dim=1.")
    
    # Ensure contiguous input for memory-efficient indexing
    x_contig = x.contiguous()
    batch, d1, d2 = x_contig.shape
    
    # Pre-allocate output tensor
    output = torch.empty((batch, d2), device=x.device, dtype=x.dtype)
    
    # Launch kernel
    fused_ext.max_dim1(x_contig, output)
    
    return output

# --- Setup for usage ---
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_init_inputs():
    return [1]

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device='cuda')
    return [x]
