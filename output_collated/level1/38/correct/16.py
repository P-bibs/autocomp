# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_022930/code_24.py
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

# ----------------------------------------------------------------------
# CUDA Source: Optimized reduction using Warp Shuffles
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Warp-level shuffle reduce
template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Optimized Kernel: Per-row sum + scaling in one go is hard due to global dependency,
// but we optimize the reduction phase heavily.
__global__ void reduce_sum_abs_kernel(const float *x, float *sum_abs, const int dim) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    float thread_sum = 0.0f;
    for (int col = tid; col < dim; col += blockDim.x) {
        thread_sum += fabsf(x[row * dim + col]);
    }

    // Shared memory for block-level reduction
    static __shared__ float sdata[32];
    int lane = tid % warpSize;
    int wid = tid / warpSize;

    // Reduce within warp
    thread_sum = warpReduceSum(thread_sum);

    // Write warp results to shared memory
    if (lane == 0) sdata[wid] = thread_sum;
    __syncthreads();

    // Reduce across warps
    float val = (tid < blockDim.x / warpSize) ? sdata[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
        if (tid == 0) sum_abs[row] = val;
    }
}

__global__ void scale_kernel(const float *x, const float *sum_abs, float *output, const int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = idx / dim;
    const int col = idx % dim;
    
    // Using grid-stride loop for scalability if needed, 
    // but here we map 1:1 for simplicity and performance
    if (row < gridDim.x) {
        output[idx] = x[idx] * (static_cast<float>(dim) / sum_abs[row]);
    }
}

void fused_normalize_cuda(torch::Tensor x, torch::Tensor sum_abs, torch::Tensor output) {
    const int batch_size = x.size(0);
    const int dim = x.size(1);
    
    // We use 256 threads (8 warps)
    const int threads = 256;
    
    reduce_sum_abs_kernel<<<batch_size, threads>>>(x.data_ptr<float>(), sum_abs.data_ptr<float>(), dim);
    
    // Kernel 2: elementwise, use enough blocks to cover total elements
    int total_elements = batch_size * dim;
    int blocks_s = (total_elements + 255) / 256;
    // Cap blocks for 2080Ti occupancy if necessary
    scale_kernel<<<blocks_s, 256>>>(x.data_ptr<float>(), sum_abs.data_ptr<float>(), output.data_ptr<float>(), dim);
}
"""

cpp_source = r"""
void fused_normalize_cuda(torch::Tensor x, torch::Tensor sum_abs, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize_cuda, "Optimized fused normalize");
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
    if not x.is_cuda:
        x = x.cuda()
    
    batch_size, dim = x.shape
    sum_abs = torch.empty(batch_size, dtype=x.dtype, device=x.device)
    output = torch.empty_like(x)
    
    fused_ext.fused_normalize(x, sum_abs, output)
    
    return output

def get_init_inputs():
    return []

def get_inputs():
    batch_size = 32768
    dim = 65535
    x = torch.rand(batch_size, dim, dtype=torch.float32).cuda()
    return [x]
