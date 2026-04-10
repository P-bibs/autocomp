# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_024918/code_2.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void fused_norm_kernel(const float* x, float* out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;

    float row_sum = 0.0f;
    const float* row_ptr = x + row * cols;

    // Coalesced memory access using float4 vectorization
    int elements_per_thread = (cols + blockDim.x * 4 - 1) / (blockDim.x * 4);
    const float4* row_ptr4 = reinterpret_cast<const float4*>(row_ptr);
    
    for (int i = 0; i < elements_per_thread; ++i) {
        int idx = i * blockDim.x + tid;
        if (idx * 4 < cols) {
            float4 vec = row_ptr4[idx];
            row_sum += fabsf(vec.x) + fabsf(vec.y) + fabsf(vec.z) + fabsf(vec.w);
        }
    }

    // Handle residual elements that don't fit into float4 chunks
    int residual_start = elements_per_thread * blockDim.x * 4;
    for (int i = residual_start + tid; i < cols; i += blockDim.x) {
        row_sum += fabsf(row_ptr[i]);
    }

    // Warp-level reduction
    row_sum = warp_reduce_sum(row_sum);

    // Store warp result contiguously in shared memory
    if (laneId == 0) {
        sdata[warpId] = row_sum;
    }
    __syncthreads();

    // Final warp reduction of warp sums
    if (warpId == 0) {
        float warp_sum = (laneId < (blockDim.x + 31) / 32) ? sdata[laneId] : 0.0f;
        warp_sum = warp_reduce_sum(warp_sum);
        if (laneId == 0) {
            sdata[0] = warp_sum / (float)cols;
        }
    }
    __syncthreads();

    // Normalize the row using the computed mean
    float mean = sdata[0];
    float4* out_row4 = reinterpret_cast<float4*>(out + row * cols);
    for (int i = 0; i < elements_per_thread; ++i) {
        int idx = i * blockDim.x + tid;
        if (idx * 4 < cols) {
            float4 vec = row_ptr4[idx];
            float4 out_vec;
            out_vec.x = vec.x / mean;
            out_vec.y = vec.y / mean;
            out_vec.z = vec.z / mean;
            out_vec.w = vec.w / mean;
            out_row4[idx] = out_vec;
        }
    }

    // Handle residual elements
    for (int i = residual_start + tid; i < cols; i += blockDim.x) {
        out[row * cols + i] = row_ptr[i] / mean;
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor out) {
    at::cuda::CUDAGuard device_guard(x.device());
    
    int rows = x.size(0);
    int cols = x.size(1);
    
    // Ensure proper alignment for float4 operations
    int threads = min(1024, ((cols + 127) / 128) * 32); // Max 1024 threads/block
    if (threads < 32) threads = 32;

    fused_norm_kernel<<<rows, threads, ((threads + 31) / 32) * sizeof(float)>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        rows, 
        cols
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op_forward", &fused_op_forward, "Fused Norm Forward with coalesced memory access");
}
"""

fused_ext = load_inline(
    name='fused_norm_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=compute_75', '-code=sm_75'],
    with_cuda=True
)

def functional_model(x):
    """Vectorized memory access + contiguous shared memory write optimization."""
    x = x.contiguous()
    out = torch.empty_like(x)
    fused_ext.fused_op_forward(x, out)
    return out

# Test parameters
batch_size = 32768
dim = 65535

def get_inputs():
    return [torch.rand(batch_size, dim, dtype=torch.float32, device='cuda')]
