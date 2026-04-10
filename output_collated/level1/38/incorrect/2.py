# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_021806/code_14.py
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
# Optimized CUDA source
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void fused_normalize_kernel(const float * __restrict__ x,
                                        float * __restrict__ output,
                                        const int batch_size,
                                        const int dim)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    float thread_sum = 0.0f;
    
    // Phase 1: Sum absolute values
    for (int col = tid; col < dim; col += blockDim.x) {
        thread_sum += fabsf(x[row * dim + col]);
    }

    // Phase 2: Warp-level reduction to compute global row sum
    // Sum across warps using warp-shuffle
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // Shared memory for block-level reduction
    __shared__ float shared_sums[32]; 
    int lane = tid % warpSize;
    int wid = tid / warpSize;

    if (lane == 0) shared_sums[wid] = thread_sum;
    __syncthreads();

    // Final sum from the first warp
    thread_sum = (tid < blockDim.x / warpSize) ? shared_sums[tid] : 0.0f;
    if (wid == 0) {
        for (int offset = blockDim.x / warpSize / 2; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
    }
    
    // Broadcast total sum to all threads
    float row_sum = __shfl_sync(0xffffffff, thread_sum, 0);

    // Phase 3: Normalize and write to output
    float scale = (float)dim / row_sum;
    for (int col = tid; col < dim; col += blockDim.x) {
        output[row * dim + col] = x[row * dim + col] * scale;
    }
}

void fused_normalize(torch::Tensor x, torch::Tensor output)
{
    const int batch_size = x.size(0);
    const int dim = x.size(1);
    const int threads = 256; 
    
    fused_normalize_kernel<<<batch_size, threads>>>(
        x.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, dim);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_normalize(torch::Tensor x, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize, "Fused normalized kernel");
}
"""

# Compile
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
    output = torch.empty_like(x)
    fused_ext.fused_normalize(x, output)
    return output

def get_init_inputs():
    return []

def get_inputs():
    return [torch.rand(32768, 65535, dtype=torch.float32)]
