# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_081907/code_16.py
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

__global__ void cumsum_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    extern __shared__ float warp_sums[];
    float carry = 0.0f;

    for (int col_start = 0; col_start < cols; col_start += blockDim.x) {
        int tid = threadIdx.x;
        int idx = col_start + tid;
        
        float val = (idx < cols) ? row_in[idx] : 0.0f;

        // 1. Intra-warp scan
        #pragma unroll
        for (int i = 1; i <= 16; i <<= 1) {
            float tmp = __shfl_up_sync(0xffffffff, val, i);
            if (tid % 32 >= i) val += tmp;
        }

        // 2. Inter-warp scan via shared memory
        int lane = tid % 32;
        int wid = tid / 32;
        if (lane == 31) warp_sums[wid] = val;
        __syncthreads();

        if (wid > 0) {
            float block_prefix = 0;
            for (int k = 0; k < wid; ++k) block_prefix += warp_sums[k];
            val += block_prefix;
        }
        __syncthreads();

        // 3. Add carry from previous blocks
        val += carry;
        if (idx < cols) row_out[idx] = val;

        // 4. Propagate carry
        if (tid == blockDim.x - 1) carry = val;
        carry = __shfl_sync(0xffffffff, carry, blockDim.x % 32 - 1 + (blockDim.x / 32) * 32); 
        // Sync carry across the whole block
        carry = __shfl_sync(0xffffffff, carry, (blockDim.x > 32 ? 31 : blockDim.x - 1));
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    const int threads = 256;
    const size_t shared_mem = (threads / 32) * sizeof(float);
    cumsum_kernel<<<rows, threads, shared_mem>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        (int)rows, 
        (int)cols
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused inclusive scan");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    original_dtype = x.dtype
    x = x.to(torch.float32)
    shape = list(x.shape)
    
    # Handle dimension transposition to ensure scan on the last dimension
    if dim != -1 and dim != x.dim() - 1:
        perm = list(range(x.dim()))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x = x.permute(perm).contiguous()
    else:
        x = x.contiguous()
    
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]
    output = torch.empty_like(x)
    
    fused_ext.fused_op(rows, cols, x, output)
    
    # Restore permutation if necessary
    if dim != -1 and dim != len(shape) - 1:
        output = output.permute(perm).contiguous()
        
    return output.to(original_dtype)
