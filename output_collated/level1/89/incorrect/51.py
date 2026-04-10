# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_083041/code_29.py
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
# CUDA kernel – optimized tiled inclusive scan with warp-level primitives
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;

__global__ void cumsum_kernel(const float* __restrict__ input,
                              float* __restrict__ output,
                              const int rows,
                              const int cols) {
    extern __shared__ float sdata[];
    
    // Each row is processed by a block. 
    // We process the row in tiles of size blockDim.x.
    const int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + (size_t)row * cols;
    float* row_out = output + (size_t)row * cols;

    float carry = 0.0f;
    for (int col_start = 0; col_start < cols; col_start += blockDim.x) {
        int tid = threadIdx.x;
        int idx = col_start + tid;
        
        // 1. Load tile into registers
        float val = (idx < cols) ? row_in[idx] : 0.0f;

        // 2. Warp-level scan
        for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
            float tmp = __shfl_up_sync(0xffffffff, val, offset);
            if (tid % WARP_SIZE >= offset) val += tmp;
        }

        // 3. Coordinate between warps using shared memory
        int warpId = tid / WARP_SIZE;
        int laneId = tid % WARP_SIZE;
        if (laneId == WARP_SIZE - 1) sdata[warpId] = val;
        __syncthreads();

        if (warpId > 0) {
            for (int i = 0; i < warpId; ++i) {
                val += sdata[i];
            }
        }
        
        // 4. Global carry propagation
        val += carry;
        
        // 5. Write out
        if (idx < cols) row_out[idx] = val;
        
        // 6. Update carry for next tile
        // Thread 255 (or last effective thread) calculates the new carry
        int last_idx = min(col_start + blockDim.x - 1, cols - 1);
        float last_val = (tid == (last_idx - col_start)) ? val : 0.0f;
        carry = __shfl_sync(0xffffffff, last_val, (last_idx - col_start));
        
        __syncthreads();
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    const int threads = 256; 
    const int warps = threads / WARP_SIZE;
    // Shared memory for warp-sums (max 8 warps for 256 threads)
    const size_t shared_mem = warps * sizeof(float);
    
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
    m.def("fused_op", &fused_op_forward, "Optimized tiled cumsum");
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
    
    # Handle dimension permutation
    permute_dims = list(range(x.dim()))
    if dim != -1 and dim != x.dim() - 1:
        permute_dims[dim], permute_dims[-1] = permute_dims[-1], permute_dims[dim]
        x = x.permute(*permute_dims)
    
    x_contig = x.contiguous()
    output = torch.empty_like(x_contig)
    
    rows = x_contig.numel() // x_contig.shape[-1]
    cols = x_contig.shape[-1]
    
    fused_ext.fused_op(rows, cols, x_contig, output)
    
    if dim != -1 and dim != x.dim() - 1:
        output = output.permute(*permute_dims)
    
    return output.to(original_dtype)
