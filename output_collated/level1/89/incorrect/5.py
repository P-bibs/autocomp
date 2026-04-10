# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_074424/code_5.py
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

# CUDA Kernel implementing a Work-Efficient Parallel Scan
# We use shared memory to cache segments of the input array.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scan_kernel(const float* __restrict__ input, float* __restrict__ output, int n) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int offset = bid * n;

    // Load data into shared memory
    temp[tid] = input[offset + tid];
    __syncthreads();

    // Up-sweep (Reduction) phase
    for (int d = n >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            int ai = (tid * 2 + 1) * (n / (2 * d)) - 1;
            int bi = (tid * 2 + 2) * (n / (2 * d)) - 1;
            temp[bi] += temp[ai];
        }
        __syncthreads();
    }

    if (tid == 0) temp[n - 1] = 0;

    // Down-sweep phase
    for (int d = 1; d < n; d <<= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = (tid * 2 + 1) * (n / (2 * d)) - 1;
            int bi = (tid * 2 + 2) * (n / (2 * d)) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Write back to global memory (adding the original element to match cumsum)
    output[offset + tid] = temp[tid] + input[offset + tid];
}

void scan_wrapper(torch::Tensor input, torch::Tensor output) {
    int batch_size = input.size(0);
    int n = input.size(1);
    // Launch one block per batch element
    scan_kernel<<<batch_size, n, n * sizeof(float)>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n
    );
}
"""

cpp_source = r"""
void scan_wrapper(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scan", &scan_wrapper, "Parallel prefix sum");
}
"""

scan_ext = load_inline(
    name='scan_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    assert dim == 1, "Only dim=1 is supported for this optimized implementation."
    output = torch.empty_like(x)
    scan_ext.scan(x, output)
    return output

# Setup for testing
batch_size = 32768
input_shape = (32768,)
dim = 1

def get_init_inputs():
    return [dim]

def get_inputs():
    return [torch.rand(batch_size, *input_shape, device='cuda')]
