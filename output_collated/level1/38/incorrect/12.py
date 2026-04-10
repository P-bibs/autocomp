# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_024918/code_8.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs L1 normalization.
    """

    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        super(ModelNew, self).__init__()

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
#include <cmath>

// Warp-level reduction utility
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Single fused kernel: reduction + scaling
__global__ void fused_normalize_kernel(const float *__restrict__ x,
                                       float *__restrict__ output,
                                       const int batch_size,
                                       const int dim)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int threads_per_block = blockDim.x; // 256
    const int warps_per_block = threads_per_block / 32;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // ========== PHASE 1: Reduction (sum of |x| per row) ==========
    float partial_sum = 0.0f;
    
    // Each thread in the warp processes multiple columns
    for (int col = lane_id; col < dim; col += 32) {
        float v = x[row * dim + col];
        partial_sum += fabsf(v);
    }
    
    // Warp-level reduction using shuffle
    float warp_sum = warp_reduce_sum(partial_sum);
    
    // Store warp sum in shared memory for block-level coordination
    __shared__ float sdata[8]; // max 8 warps per block with 256 threads
    if (lane_id == 0) {
        sdata[warp_id] = warp_sum;
    }
    __syncthreads();
    
    // Accumulate all warp sums into one (using first warp)
    float row_sum = 0.0f;
    if (warp_id == 0) {
        row_sum = (lane_id < warps_per_block) ? sdata[lane_id] : 0.0f;
        row_sum = warp_reduce_sum(row_sum);
    }
    __syncthreads();
    
    // Store final sum in shared memory for access by all threads
    if (tid == 0) {
        sdata[0] = row_sum;
    }
    __syncthreads();
    
    float sum_abs = sdata[0];
    
    // ========== PHASE 2: Scaling (vectorized) ==========
    // All threads participate in writing output
    // Vectorize by processing float4
    
    const int elements_per_float4 = 4;
    const int float4_dim = dim / elements_per_float4;
    const float scale_factor = static_cast<float>(dim) / sum_abs;
    
    // Use vectorized access: process as float4*
    const float4 *x_vec = (const float4 *)x;
    float4 *out_vec = (float4 *)output;
    
    // Each thread processes multiple float4 values
    for (int vec_idx = tid; vec_idx < float4_dim; vec_idx += threads_per_block) {
        int global_vec_idx = row * float4_dim + vec_idx;
        
        // Load float4
        float4 data = x_vec[global_vec_idx];
        
        // Scale each float in the float4
        data.x *= scale_factor;
        data.y *= scale_factor;
        data.z *= scale_factor;
        data.w *= scale_factor;
        
        // Store float4
        out_vec[global_vec_idx] = data;
    }
    
    // Handle remainder elements if dim is not divisible by 4
    if ((dim % 4) > 0) {
        int start_col = float4_dim * 4;
        for (int col = start_col + tid; col < dim; col += threads_per_block) {
            int global_idx = row * dim + col;
            output[global_idx] = x[global_idx] * scale_factor;
        }
    }
}

void fused_normalize(torch::Tensor x,
                     torch::Tensor output)
{
    const int batch_size = x.size(0);
    const int dim        = x.size(1);
    const int threads    = 256;  // 8 warps per block
    const int blocks     = batch_size;

    fused_normalize_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, dim);
    
    // Single synchronization at the end
    cudaDeviceSynchronize();
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_normalize(torch::Tensor x,
                     torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize,
          "Fused vectorized abs-reduction + normalization kernel");
}
"""

fused_ext = load_inline(
    name='fused_norm_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)


def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize each row of x by the mean of its absolute values.
    Equivalent to: x / torch.mean(torch.abs(x), dim=1, keepdim=True)
    """
    if not x.is_cuda:
        x = x.cuda()

    batch_size = x.size(0)
    dim = x.size(1)

    output = torch.empty_like(x)
    fused_ext.fused_normalize(x, output)

    return output


def get_init_inputs():
    return []

def get_inputs():
    batch_size = 32768
    dim = 65535
    x = torch.rand(batch_size, dim, dtype=torch.float32)
    return [x]
