# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_080721/code_22.py
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
# CUDA Kernel: Optimized Inclusive Scan
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cumsum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols)
{
    extern __shared__ float sdata[];
    const int threads = blockDim.x;
    const int warps = threads / 32;

    const int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    float carry = 0.0f;

    for (int col_start = 0; col_start < cols; col_start += threads) {
        int tid = threadIdx.x;
        int idx = col_start + tid;

        // Load tile to shared memory
        sdata[tid] = (idx < cols) ? row_in[idx] : 0.0f;
        __syncthreads();

        // 1. Warp-level scan using shuffle
        float val = sdata[tid];
        int lane = tid & 31;
        for (int offset = 1; offset < 32; offset <<= 1) {
            float tmp = __shfl_up_sync(0xffffffff, val, offset);
            if (lane >= offset) val += tmp;
        }
        
        // 2. Store warp totals in shared memory
        int warp_id = tid >> 5;
        if (lane == 31) {
            sdata[threads + warp_id] = val;
        }
        __syncthreads();

        // 3. Scan warp totals (serial scan in thread 0)
        if (tid < warps) {
            float warp_sum = sdata[threads + tid];
            float scan = 0.0f;
            for (int i = 0; i < tid; ++i) scan += sdata[threads + i];
            sdata[threads + tid] = warp_sum + scan;
        }
        __syncthreads();

        // 4. Incorporate carry and global carry
        float block_carry = (warp_id > 0) ? sdata[threads + warp_id - 1] : 0.0f;
        float final_val = val + block_carry + carry;

        if (idx < cols) {
            row_out[idx] = final_val;
        }
        
        // Update carry for next tile
        if (tid == threads - 1) {
            carry = final_val;
        }
        __syncthreads();
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    const int threads = 256;
    const int blocks = static_cast<int>(rows);
    const size_t shared_mem = (threads + (threads / 32)) * sizeof(float);
    
    cumsum_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        static_cast<int>(rows), 
        static_cast<int>(cols)
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
    
    # Alignment: ensure scan dimension is last
    permute_dims = None
    if dim != -1 and dim != x.dim() - 1:
        permute_dims = list(range(x.dim()))
        permute_dims[dim], permute_dims[-1] = permute_dims[-1], permute_dims[dim]
        x = x.permute(*permute_dims)
    
    x = x.contiguous()
    output = torch.empty_like(x)
    
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]
    
    fused_ext.fused_op(rows, cols, x, output)
    
    # Restore original shape
    if permute_dims is not None:
        output = output.permute(*permute_dims)
    
    return output.to(original_dtype)
