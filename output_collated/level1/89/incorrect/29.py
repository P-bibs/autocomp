# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_080721/code_14.py
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

# The original implementation uses a naive O(N^2) serial scan inside the kernel.
# For high performance with large vectors, we implement a vectorized device-level 
# scan kernel. Given the constraint of a single kernel for the operation, we 
# use a block-wide prefix sum approach.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// A block-level parallel prefix sum (Blelloch scan) implementation.
// This handles each row of the input independently.
__global__ void cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    extern __shared__ float smem[];
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    // Load data from global to shared
    if (tid < N) {
        smem[tid] = input[bid * N + tid];
    } else {
        smem[tid] = 0.0f;
    }
    __syncthreads();

    // Up-sweep (reduce)
    for (int stride = 1; stride < N; stride <<= 1) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < N) {
            smem[idx] += smem[idx - stride];
        }
        __syncthreads();
    }

    // Down-sweep
    if (tid == N - 1) smem[tid] = 0;
    __syncthreads();

    for (int stride = N / 2; stride > 0; stride >>= 1) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < N) {
            float t = smem[idx];
            smem[idx] += smem[idx - stride];
            smem[idx - stride] = t;
        }
        __syncthreads();
    }

    // Write back
    if (tid < N) {
        output[bid * N + tid] = smem[tid] + input[bid * N + tid];
    }
}

void launch_cumsum(torch::Tensor input, torch::Tensor output) {
    int batch_size = input.size(0);
    int dim_size = input.size(1);
    
    // Using blocks to handle rows, threads to handle the scan of the dim_size
    // This assumes dim_size <= 1024 (max threads per block)
    int threads = dim_size;
    int shared_mem = dim_size * sizeof(float);
    
    cumsum_kernel<<<batch_size, threads, shared_mem>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        dim_size
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_cumsum(torch::Tensor input, torch::Tensor output);

torch::Tensor fused_cumsum(torch::Tensor input, int dim) {
    auto output = torch::empty_like(input);
    launch_cumsum(input, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_cumsum", &fused_cumsum, "Fused Cumulative Sum Kernel");
}
"""

# Compile the optimized kernel
fused_ext = load_inline(
    name='fused_cumsum_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

batch_size = 32768
input_shape = (32768,)
dim = 1

def functional_model(x, *, dim):
    """
    Optimized functional model using a custom Blelloch scan CUDA kernel.
    """
    return fused_ext.fused_cumsum(x, dim)

def get_init_inputs():
    return [dim]

def get_inputs():
    return [torch.rand(batch_size, *input_shape).cuda()]
