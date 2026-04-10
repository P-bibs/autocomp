# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_081907/code_15.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['dim']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    A simple model that performs a cumulative sum (prefix sum) operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the scan operation.
    """

    def __init__(self, dim):
        """
        Initialize the Scan model.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
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

# The optimized kernel uses memory coalescing. 
# While the previous implementation was row-major (which caused poor occupancy),
# this implementation processes columns in parallel. Since the layout is contiguous,
# adjacent threads in a warp access adjacent memory addresses, maximizing L1/L2 cache utilization.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check for column-parallel assignment
    if (col >= cols) return;

    // Perform cumulative sum down the column
    // This access pattern is coalesced as consecutive threads process the same row offset
    float running_sum = 0.0f;
    for (int row = 0; row < rows; ++row) {
        int idx = row * cols + col;
        running_sum += input[idx];
        output[idx] = running_sum;
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    // 256 threads is generally a good balance for occupancy on RTX 2080Ti
    const int threads = 256;
    const int blocks = (static_cast<int>(cols) + threads - 1) / threads;
    
    cumsum_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        static_cast<int>(rows), 
        static_cast<int>(cols)
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Coalesced fused cumsum kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    """
    Computes cumulative sum along a specified dimension using a custom CUDA kernel.
    The kernel is optimized for coalesced memory access by parallelizing over columns.
    """
    original_dtype = x.dtype
    x = x.to(torch.float32)
    
    # Handle arbitrary dimensions by permuting
    if dim != -1 and dim != x.dim() - 1:
        permute_dims = list(range(x.dim()))
        permute_dims[dim], permute_dims[-1] = permute_dims[-1], permute_dims[dim]
        # Move target dimension to the last axis for coalesced kernel access
        x = x.permute(*permute_dims)
    
    # Ensure memory is contiguous for clean row-major strides
    x_contig = x.contiguous()
    output = torch.empty_like(x_contig)
    
    # Flatten everything except the target dimension to treat as rows x cols
    flat_shape = x_contig.shape
    rows = x_contig.numel() // flat_shape[-1]
    cols = flat_shape[-1]
    
    # Run the kernel
    fused_ext.fused_op(rows, cols, x_contig, output)
    
    # If permuted, transpose result back to original shape
    if dim != -1 and dim != x.dim() - 1:
        output = output.permute(*permute_dims)
    
    return output.to(original_dtype)
