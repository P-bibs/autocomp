# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_074424/code_16.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Optimized Exclusive Scan using Warp Shuffles
__device__ __forceinline__ float warp_scan(float val) {
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1) {
        float tmp = __shfl_up_sync(0xFFFFFFFF, val, i);
        if (threadIdx.x % 32 >= i) val += tmp;
    }
    return val;
}

__global__ void cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    extern __shared__ float shared_sums[];
    float carry = 0.0f;

    for (int col_start = 0; col_start < cols; col_start += blockDim.x) {
        int idx = col_start + threadIdx.x;
        float val = (idx < cols) ? row_in[idx] : 0.0f;

        // 1. Warp-level scan
        float scan_val = warp_scan(val);

        // 2. Reduce warps and store in shared memory
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;
        if (lane == 31) shared_sums[wid] = scan_val;
        __syncthreads();

        // 3. First warp scans the block-partials
        if (threadIdx.x < 32) {
            float partial = (threadIdx.x < (blockDim.x / 32)) ? shared_sums[threadIdx.x] : 0.0f;
            float scan_p = warp_scan(partial);
            shared_sums[threadIdx.x] = scan_p;
        }
        __syncthreads();

        // 4. Add block-level carry to each thread
        float block_carry = (wid > 0) ? shared_sums[wid - 1] : 0.0f;
        row_out[idx] = scan_val + block_carry + carry;

        // 5. Update carry for next tile
        __syncthreads();
        if (threadIdx.x == blockDim.x - 1) {
            carry += row_out[idx];
        }
        carry = __shfl_sync(0xFFFFFFFF, carry, blockDim.x - 1);
        __syncthreads();
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    const int threads = 256;
    const int shared_mem = (threads / 32) * sizeof(float);
    cumsum_kernel<<<rows, threads, shared_mem>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        (int)rows, 
        (int)cols
    );
}
"""

cpp_source = r"""
void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Optimized cumsum");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3'],
    with_cuda=True
)

def functional_model(x, *, dim=-1):
    original_dtype = x.dtype
    x = x.contiguous().to(torch.float32)
    
    if dim != -1 and dim != x.dim() - 1:
        raise NotImplementedError("Kernel optimized for the last dimension.")
        
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]
    output = torch.empty_like(x)
    
    fused_ext.fused_op(rows, cols, x, output)
    return output.to(original_dtype)
