# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_081907/code_20.py
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

# The problem: perform cumsum along dim=1. Given input_shape=(32768,), 
# each row of the (batch_size, 32768) matrix is independent.
# We implement a parallel prefix sum using warp-level primitives.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_scan(float val) {
    for (int offset = 1; offset < 32; offset <<= 1) {
        float x = __shfl_up_sync(0xffffffff, val, offset);
        if (threadIdx.x >= offset) val += x;
    }
    return val;
}

__global__ void cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    
    // Each block handles one row (32768 elements)
    // We use a shared memory buffer for block-level scan
    extern __shared__ float sdata[];
    
    for (int j = col; j < N; j += blockDim.x) {
        sdata[j] = input[row * N + j];
    }
    __syncthreads();

    // Perform scan in blocks of 32 (warp size)
    // 1. Warp level scan
    for (int j = col; j < N; j += blockDim.x) {
        float val = sdata[j];
        int warp_id = (j % 32);
        
        float scan = val;
        for (int offset = 1; offset < 32; offset <<= 1) {
            float x = __shfl_up_sync(0xffffffff, scan, offset);
            if (warp_id >= offset) scan += x;
        }
        sdata[j] = scan;
    }
    __syncthreads();

    // 2. Aggregate warp results (Add last element of previous warp to current warp)
    // For 32768 elements and 1024 threads, this is efficient
    if (col < (N / 32)) {
        float sum = 0;
        for (int w = 0; w < (N / 32); ++w) {
            float last_val = sdata[w * 32 + 31];
            if (col > w) sum += last_val;
        }
        for (int k = 0; k < 32; ++k) {
            if (col * 32 + k < N) sdata[col * 32 + k] += sum;
        }
    }
    __syncthreads();

    for (int j = col; j < N; j += blockDim.x) {
        output[row * N + j] = sdata[j];
    }
}

void launch_cumsum(torch::Tensor input, torch::Tensor output) {
    int batch_size = input.size(0);
    int N = input.size(1);
    
    // We use 1024 threads per block to cover the row
    int threads = 1024;
    dim3 blocks(batch_size);
    
    cumsum_kernel<<<blocks, threads, N * sizeof(float)>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        N
    );
}
"""

cpp_source = r"""
void launch_cumsum(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cumsum", &launch_cumsum, "Custom CUDA cumsum");
}
"""

# Compile the extension
cumsum_ext = load_inline(
    name='cumsum_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    # Requirement: dim=1
    output = torch.empty_like(x)
    cumsum_ext.cumsum(x, output)
    return output

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_init_inputs():
    return [dim]

def get_inputs():
    return [torch.rand(batch_size, *input_shape, device='cuda', dtype=torch.float32)]
