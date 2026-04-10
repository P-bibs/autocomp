# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_152339/code_5.py
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

# The implementation focuses on maximizing throughput by performing a 
# warp-level reduction. Since dim2 (4095) is large, we iterate over the 
# row and then perform a final warp-level reduction on the partial results.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void max_reduce_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                  int batch, int dim1, int dim2) {
    // Each block handles one slice (b, d)
    // Using 32 threads per block to allow for warp-level operations
    int b = blockIdx.x / dim1;
    int d = blockIdx.x % dim1;
    
    const float* row = input + (b * dim1 * dim2) + (d * dim2);
    
    float local_max = -1e38f; // Initialize with sufficiently small value
    
    // Grid-stride loop to handle cases where dim2 > total threads
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        local_max = fmaxf(local_max, row[i]);
    }

    // Warp-level reduction to combine values across the block
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    }

    // The first thread writes the result for the entire row
    if (threadIdx.x == 0) {
        output[b * dim1 + d] = local_max;
    }
}

void max_reduce_forward(torch::Tensor input, torch::Tensor output, int dim) {
    int batch = input.size(0);
    int dim1 = input.size(1);
    int dim2 = input.size(2);
    
    // We target performance for dim=2 (last dim)
    int blocks = batch * dim1;
    int threads = 32; 
    
    max_reduce_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), 
        batch, dim1, dim2
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void max_reduce_forward(torch::Tensor input, torch::Tensor output, int dim);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reduce_forward", &max_reduce_forward, "Max reduce forward (CUDA)");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='max_reduce_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    # Enforce input criteria
    if dim != 2:
        return torch.max(x, dim=dim)[0]
    
    out_shape = list(x.shape)
    out_shape.pop(dim)
    output = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    
    # We call the custom CUDA kernel
    # Input x is expected to be contiguous for coalesced access
    fused_ext.max_reduce_forward(x.contiguous(), output, dim)
    return output

# Verification/Setup Logic
batch_size, dim1, dim2 = 128, 4096, 4095

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device='cuda', dtype=torch.float32)
    return [x]
