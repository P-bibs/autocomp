# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_083041/code_28.py
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

extern "C" __global__ void cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
    // 2D grid strategy: y-dim for rows, x-dim for tiling columns
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    // Shared memory: warps per block is 8 (256 threads)
    __shared__ float s_warp_sums[8];
    float thread_carry = 0.0f;

    for (int col_start = 0; col_start < cols; col_start += 256) {
        int lane = threadIdx.x % 32;
        int warp_id = threadIdx.x / 32;
        int idx = col_start + threadIdx.x;
        
        float val = (idx < cols) ? row_in[idx] : 0.0f;

        // Warp scan
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            float tmp = __shfl_up_sync(0xFFFFFFFF, val, offset);
            if (lane >= offset) val += tmp;
        }

        // Warp sum exchange
        if (lane == 31) s_warp_sums[warp_id] = val;
        __syncthreads();

        if (threadIdx.x < 8) {
            float w_sum = s_warp_sums[threadIdx.x];
            for (int i = 0; i < threadIdx.x; ++i) w_sum += s_warp_sums[i];
            s_warp_sums[threadIdx.x] = w_sum;
        }
        __syncthreads();

        float carry = (warp_id == 0) ? thread_carry : s_warp_sums[warp_id - 1];
        val += carry;

        if (idx < cols) row_out[idx] = val;

        if (threadIdx.x == 255) thread_carry += s_warp_sums[7];
        __syncthreads();
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    // 256 threads per block, 2D grid logic to handle massive row counts
    dim3 threads(256, 1);
    dim3 blocks(1, (rows + threads.y - 1) / threads.y);
    cumsum_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), (int)rows, (int)cols);
}
"""

cpp_source = r"""
void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fused_op", &fused_op_forward, "Optimized Cumsum"); }
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    shape = list(x.shape)
    
    # Handle arbitrary dimensions by reshaping to 2D
    if dim == -1: dim = x.ndim - 1
    
    perm = list(range(x.ndim))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    x = x.permute(*perm).contiguous()
    
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]
    output = torch.empty_like(x)
    
    fused_ext.fused_op(rows, cols, x, output)
    
    # Invert permutation
    output = output.permute(*perm)
    return output.to(orig_dtype)
