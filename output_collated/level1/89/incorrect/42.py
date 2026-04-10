# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_083041/code_5.py
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

__global__ void cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    // Use shared memory to store warp sums for inter-warp communication
    extern __shared__ float warp_sums[];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = (blockDim.x + 31) / 32;

    float carry = 0.0f;
    for (int col_start = 0; col_start < cols; col_start += blockDim.x) {
        int idx = col_start + tid;
        float val = (idx < cols) ? row_in[idx] : 0.0f;

        // Warp-level inclusive scan using shuffle primitives
        for (int offset = 1; offset < 32; offset <<= 1) {
            float n = __shfl_up_sync(0xFFFFFFFF, val, offset);
            if (lane_id >= offset) val += n;
        }

        // Store the warp sum (last element in each warp) in shared memory
        if (lane_id == 31) {
            warp_sums[warp_id] = val;
        }
        __syncthreads();

        // Scan the warp sums to compute inter-warp carries
        if (warp_id == 0) {
            float warp_val = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
            for (int offset = 1; offset < 32; offset <<= 1) {
                float n = __shfl_up_sync(0xFFFFFFFF, warp_val, offset);
                if (lane_id >= offset) warp_val += n;
            }
            if (lane_id < num_warps) warp_sums[lane_id] = warp_val;
        }
        __syncthreads();

        // Add the carry from previous warps
        if (warp_id > 0) {
            val += warp_sums[warp_id - 1];
        }

        // Add the carry from previous blocks
        val += carry;

        if (idx < cols) row_out[idx] = val;

        // Update carry for next iteration with the last element of this block
        if (tid == blockDim.x - 1) {
            carry = val;
        }
        __syncthreads();
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    const int threads = 256;
    const int num_warps = (threads + 31) / 32;
    const size_t shared_mem = num_warps * sizeof(float);
    dim3 grid(rows);
    cumsum_kernel<<<grid, threads, shared_mem>>>(
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
    m.def("fused_op", &fused_op_forward, "Fused cumsum execution");
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
    
    if dim != -1 and dim != x.dim() - 1:
        permute_dims = list(range(x.dim()))
        permute_dims[dim], permute_dims[-1] = permute_dims[-1], permute_dims[dim]
        x = x.permute(*permute_dims)
    
    x = x.contiguous()
    output = torch.empty_like(x)
    
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]
    
    fused_ext.fused_op(rows, cols, x, output)
    
    if dim != -1 and dim != x.dim() - 1:
        output = output.permute(*permute_dims)
    
    return output.to(original_dtype)
