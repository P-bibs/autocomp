# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_015650/code_8.py
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
# Optimized CUDA Source
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fused_normalize_kernel(const float* __restrict__ x,
                                       float* __restrict__ output,
                                       const int batch_size,
                                       const int dim)
{
    const int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_ptr = x + row * dim;
    float* out_ptr = output + row * dim;

    // Phase 1: Grid-stride reduction for sum of absolute values
    float sum = 0.0f;
    for (int col = threadIdx.x; col < dim; col += blockDim.x) {
        sum += fabsf(row_ptr[col]);
    }

    // Phase 2: Warp-level reduction
    // Using shuffle primitives for faster reduction without shared memory
    float val = sum;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    // Broadcast the sum back to all threads in the block
    // We use a single shared memory location for the row sum
    __shared__ float row_sum;
    if (threadIdx.x == 0) {
        row_sum = val;
    }
    __syncthreads();
    
    const float final_sum = row_sum;
    const float scale = static_cast<float>(dim) / final_sum;

    // Phase 3: Scaling (Normalization)
    for (int col = threadIdx.x; col < dim; col += blockDim.x) {
        out_ptr[col] = row_ptr[col] * scale;
    }
}

void fused_normalize(torch::Tensor x, torch::Tensor output) {
    const int batch_size = x.size(0);
    const int dim = x.size(1);
    // 256 threads is optimal for most GPU architectures including 2080Ti
    const int threads = 256;
    
    fused_normalize_kernel<<<batch_size, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_normalize(torch::Tensor x, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize, "Fused normalized kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_normalize_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized normalization: one kernel launch, warp reductions,
    zero intermediate global memory buffers.
    """
    if not x.is_cuda:
        x = x.cuda()

    output = torch.empty_like(x)
    # The kernel is launched with 1 block per row, ideal for large rows
    fused_ext.fused_normalize(x, output)
    return output

def get_init_inputs():
    return []

def get_inputs():
    batch_size = 32768
    dim = 65535
    x = torch.rand(batch_size, dim, dtype=torch.float32, device='cuda')
    return [x]
