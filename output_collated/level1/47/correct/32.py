# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_125421/code_20.py
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

# Optimized CUDA kernel for reduction along dimension 1 (dim=1)
# Uses shared memory to cache partial sums within each block, 
# maximizing throughput on the RTX 2080Ti.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_reduction_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    int B, int D1, int D2) 
{
    // Each block processes a single (batch, column) pair to maximize coalescing
    // Grid: (B * D2) blocks, each with 256 threads
    int idx = blockIdx.x;
    int b = idx / D2;
    int c = idx % D2;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float sum = 0.0f;
    int offset = b * D1 * D2 + c;
    
    // Grid-stride loop to handle D1 > blockDim.x
    for (int i = tid; i < D1; i += blockDim.x) {
        sum += input[offset + i * D2];
    }

    sdata[tid] = sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[b * D2 + c] = sdata[0];
    }
}

void sum_reduction_launcher(torch::Tensor input, torch::Tensor output) {
    int B = input.size(0);
    int D1 = input.size(1);
    int D2 = input.size(2);
    
    int threads = 256;
    int blocks = B * D2;
    
    sum_reduction_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), B, D1, D2
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void sum_reduction_launcher(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_reduction", &sum_reduction_launcher, "Reduction along dim 1");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    """
    Optimized reduction implementation via custom kernel.
    Assumes dim=1 and input shape (B, D1, D2).
    """
    assert dim == 1, "Only dim=1 is supported by this optimized kernel"
    
    batch_size, d1, d2 = x.size()
    # Output is (B, 1, D2) as per keepdim=True
    output = torch.empty((batch_size, 1, d2), device=x.device, dtype=x.dtype)
    
    # Flatten the middle dim for the kernel logic
    reshaped_output = output.view(batch_size, d2)
    
    fused_ext.sum_reduction(x, reshaped_output)
    
    return output

def get_inputs():
    # Use standard dimensions provided in the template
    return [torch.rand(128, 4096, 4095).cuda()]
