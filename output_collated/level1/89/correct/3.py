# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_072553/code_5.py
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

# The problem size (32768, 32768) is too large for the 48KB/96KB shared memory limit
# per block (a float array of 32768 is 128KB). 
# We implement a tiled scan approach: scan chunks of 1024 elements in shared memory,
# prefixing the sum of the previous tile to the current one.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE_SIZE 1024

__global__ void scan_tiled_kernel(const float* __restrict__ input, float* __restrict__ output, int n) {
    extern __shared__ float temp[];
    int bid = blockIdx.x;
    const float* row_in = input + bid * n;
    float* row_out = output + bid * n;

    float running_sum = 0.0f;
    for (int tile = 0; tile < n; tile += TILE_SIZE) {
        int tid = threadIdx.x;
        int idx = tile + tid;

        // Load tile
        if (idx < n) temp[tid] = row_in[idx];
        else temp[tid] = 0.0f;
        __syncthreads();

        // Blelloch Scan in shared memory
        for (int stride = 1; stride < TILE_SIZE; stride *= 2) {
            int pos = (tid + 1) * stride * 2 - 1;
            if (pos < TILE_SIZE) temp[pos] += temp[pos - stride];
            __syncthreads();
        }

        for (int stride = TILE_SIZE / 2; stride > 0; stride /= 2) {
            int pos = (tid + 1) * stride * 2 - 1;
            if (pos + stride < TILE_SIZE) temp[pos + stride] += temp[pos];
            __syncthreads();
        }

        // Write output and update global running sum
        if (idx < n) {
            row_out[idx] = temp[tid] + running_sum;
        }
        
        // Broadcast the last element of the tile to all threads for the next loop
        __syncthreads();
        float last_val = temp[TILE_SIZE - 1];
        __syncthreads();
        running_sum += last_val;
    }
}

void scan_wrapper(torch::Tensor input, torch::Tensor output) {
    int batch = input.size(0);
    int n = input.size(1);
    // Launch blocks equal to rows, each using 1024 threads
    scan_tiled_kernel<<<batch, TILE_SIZE, TILE_SIZE * sizeof(float)>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void scan_wrapper(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scan_wrapper", &scan_wrapper, "Parallel Prefix Sum (Tiled)");
}
"""

scan_module = load_inline(
    name='scan_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim=1):
    output = torch.empty_like(x)
    scan_module.scan_wrapper(x, output)
    return output

# Evaluation helpers
batch_size = 32768
input_shape = (32768,)
dim = 1

def get_init_inputs():
    return [dim]

def get_inputs():
    # Returning a tensor on GPU as required by the custom kernel
    return [torch.rand(32768, 32768, device='cuda', dtype=torch.float32)]
