# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_074424/code_22.py
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

# ----------------------------------------------------------------------
#  Optimized CUDA kernel
#  Uses warp-level shuffle instructions for scan (Kogge-Stone) and
#  avoids excessive __syncthreads(). 
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
    extern __shared__ float sdata[];
    // sdata stores the partial sums of each warp to be shared across the block
    float* warp_sums = sdata;

    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + (size_t)row * cols;
    float* row_out = output + (size_t)row * cols;

    float carry = 0.0f;

    for (int col_start = 0; col_start < cols; col_start += BLOCK_SIZE) {
        int tid = threadIdx.x;
        int idx = col_start + tid;

        // Load data
        float val = (idx < cols) ? row_in[idx] : 0.0f;

        // Warp-level inclusive scan (shuffle)
        for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
            float tmp = __shfl_up_sync(0xffffffff, val, offset);
            if (tid % WARP_SIZE >= offset) val += tmp;
        }

        // Store warp results in shared memory
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;
        if (lane_id == WARP_SIZE - 1) {
            warp_sums[warp_id] = val;
        }
        __syncthreads();

        // Add prefix from previous warps
        if (warp_id > 0) {
            float prefix = 0.0f;
            for (int i = 0; i < warp_id; ++i) {
                prefix += warp_sums[i];
            }
            val += prefix;
        }

        // Add carry from previous block
        val += carry;

        // Write output
        if (idx < cols) row_out[idx] = val;

        // Update carry for the next tile
        carry = __shfl_sync(0xffffffff, val, BLOCK_SIZE - 1);
        __syncthreads();
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    const int threads = BLOCK_SIZE;
    const int blocks = (int)rows;
    // Shared memory for warp sums (up to 256/32 = 8 floats)
    size_t shared_mem = (BLOCK_SIZE / WARP_SIZE) * sizeof(float);
    
    cumsum_kernel<<<blocks, threads, shared_mem>>>(
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
    m.def("fused_op", &fused_op_forward, "Fused cumsum execution forward");
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
    # Ensure float32 for uniform processing
    x = x.to(torch.float32)
    
    # Handle dimension permute to ensure last dim is the scan dim
    original_shape = list(x.shape)
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
    
    # Restore shape
    if dim != -1 and dim != len(original_shape) - 1:
        output = output.permute(perm).contiguous()
    
    return output.to(original_dtype)
