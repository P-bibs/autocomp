# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_032736/code_17.py
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

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void fused_normalize_kernel(const float* __restrict__ x,
                                       float* __restrict__ output,
                                       const int dim)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Step 1: Reduce sum of absolutes
    float thread_sum = 0.0f;
    for (int col = tid; col < dim; col += blockDim.x) {
        thread_sum += fabsf(x[row * dim + col]);
    }

    // Warp-level reduction
    thread_sum = warp_reduce_sum(thread_sum);

    // Block-level reduction
    __shared__ float shared_sums[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) shared_sums[warp_id] = thread_sum;
    __syncthreads();

    // Final result in shared memory
    if (tid < ((blockDim.x + 31) / 32)) {
        thread_sum = shared_sums[tid];
    } else {
        thread_sum = 0.0f;
    }
    
    if (warp_id == 0) {
        thread_sum = warp_reduce_sum(thread_sum);
        if (lane_id == 0) shared_sums[0] = thread_sum;
    }
    __syncthreads();

    const float row_sum = shared_sums[0];
    const float inv_sum = (row_sum > 0.0f) ? (static_cast<float>(dim) / row_sum) : 0.0f;

    // Step 2: Scale
    for (int col = tid; col < dim; col += blockDim.x) {
        output[row * dim + col] = x[row * dim + col] * inv_sum;
    }
}

void fused_normalize(torch::Tensor x, torch::Tensor output) {
    const int batch_size = x.size(0);
    const int dim = x.size(1);
    // 256 threads is generally optimal for memory-bound tasks on 2080Ti
    const int threads = 256; 
    fused_normalize_kernel<<<batch_size, threads>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), dim
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_normalize(torch::Tensor x, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize, "Fused abs-reduction and scaling");
}
"""

fused_ext = load_inline(
    name='fused_norm_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        x = x.cuda()
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
