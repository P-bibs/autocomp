# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_151147/code_5.py
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

# Optimized CUDA kernel using warp-level primitives for coalescence and throughput.
# Each warp acts as a cooperative group to perform the reduction on a specific row.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void max_reducer_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                   int total_rows, int dim2) {
    // Each block processes a range of rows.
    // Each warp (32 threads) within the block handles one row.
    int row = blockIdx.x * (blockDim.y / 32) + (threadIdx.y / 32);
    if (row >= total_rows) return;

    int lane = threadIdx.y % 32;
    const float* row_ptr = input + (long long)row * dim2;
    
    float max_val = -1e38f; 

    // Strided access: threads within the warp work together reading chunks of size 32
    for (int i = lane; i < dim2; i += 32) {
        max_val = fmaxf(max_val, row_ptr[i]);
    }

    // Warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    // Lane 0 of the warp writes the result for the row
    if (lane == 0) {
        output[row] = max_val;
    }
}

void max_reducer(torch::Tensor input, torch::Tensor output) {
    int total_rows = input.size(0) * input.size(1);
    int dim2 = input.size(2);
    
    // Each block uses 128 threads (4 warps of 32 threads)
    // 4 rows per block. 
    int threads_per_warp = 32;
    int warps_per_block = 4;
    dim3 threads(1, warps_per_block * threads_per_warp);
    int blocks = (total_rows + warps_per_block - 1) / warps_per_block;
    
    max_reducer_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), 
                                            total_rows, dim2);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_reducer(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reducer", &max_reducer, "Coalesced row-wise max reduction");
}
"""

# Compile the JIT extension
fused_ext = load_inline(
    name='max_reduction',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    """
    Optimized max reduction along dim=2 for (batch_size, dim1, dim2).
    Ensures optimal memory throughput via warp-level coalesced reduction.
    """
    # The evaluation assumes x is moved to the device by the setup
    if dim != 2:
        return torch.max(x, dim=dim)[0]
    
    output = torch.empty((x.shape[0], x.shape[1]), device=x.device, dtype=x.dtype)
    fused_ext.max_reducer(x.contiguous(), output)
    return output

# --- Evaluation setup variables ---
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device='cuda')
    return [x]
