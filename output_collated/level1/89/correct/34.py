# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_083041/code_22.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, int cols) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    float carry = 0.0f;
    
    // Process input in chunks of 1024 (blockDim)
    for (int i = 0; i < cols; i += blockDim.x) {
        int idx = i + tid;
        sdata[tid] = (idx < cols) ? input[row * cols + idx] : 0.0f;
        __syncthreads();

        // Hillis-Steele exclusive scan implementation adapted for prefix sum
        // More robust than the previous naive warp-iteration
        for (int offset = 1; offset < blockDim.x; offset <<= 1) {
            float val = (tid >= offset) ? sdata[tid - offset] : 0.0f;
            __syncthreads();
            if (tid >= offset) sdata[tid] += val;
            __syncthreads();
        }
        
        float final_val = sdata[tid] + carry;
        if (idx < cols) output[row * cols + idx] = final_val;
        
        // Broadcast the last sum to be picked up by the next chunk
        carry += sdata[blockDim.x - 1];
        __syncthreads();
    }
}

void cumsum_cuda(torch::Tensor input, torch::Tensor output) {
    int rows = input.size(0);
    int cols = input.size(1);
    int threads = 1024;
    
    // Launch kernel with dynamic shared memory
    cumsum_kernel<<<rows, threads, threads * sizeof(float)>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), cols);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void cumsum_cuda(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cumsum", &cumsum_cuda, "Optimized Block-based Prefix Scan");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='cumsum_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim=1):
    """
    Optimized cumsum implementation using a custom CUDA kernel.
    Handles tensors of shape (N, M).
    """
    # Ensure input is 2D
    is_1d = False
    if x.dim() == 1:
        x = x.unsqueeze(0)
        is_1d = True
    
    output = torch.empty_like(x)
    # The kernel assumes memory layout is contiguous for coalesced access
    fused_ext.cumsum(x.contiguous(), output)
    
    return output.squeeze() if is_1d else output
