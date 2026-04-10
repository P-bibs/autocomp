# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_021806/code_11.py
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

# Optimization: Fused kernel to perform ABS, Mean, and Division in a single pass.
# Uses block-level reduction with atomic-like behavior implicitly via shared memory.
# Note: For very high dimensions, the reduction loop ensures correctness.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_norm_kernel(const float* x, float* out, int cols) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    const float* row_ptr = x + (size_t)row * cols;
    float* out_ptr = out + (size_t)row * cols;

    float row_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_sum += fabsf(row_ptr[i]);
    }

    sdata[threadIdx.x] = row_sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    float mean = sdata[0] / (float)cols;
    
    // Final pass to calculate output
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_ptr[i] = row_ptr[i] / mean;
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor out) {
    int rows = x.size(0);
    int cols = x.size(1);
    
    // Block size selection: 512 is common for RTX 2080 Ti
    int threads = 512;
    // We launch 'rows' blocks. This is efficient as long as rows is high.
    fused_norm_kernel<<<rows, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), cols
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor x, torch::Tensor out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Abs-Mean-Div operation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    Optimized implementation:
    1. Assumes input is contiguous CUDA float32 tensor.
    2. Uses fused CUDA kernel to avoid intermediate allocations and global memory overhead.
    """
    if not x.is_cuda:
        x = x.cuda()
    if not x.is_contiguous():
        x = x.contiguous()
    
    out = torch.empty_like(x)
    fused_ext.fused_op(x, out)
    return out

# Input configuration
batch_size = 32768
dim = 65535

def get_init_inputs():
    return []

def get_inputs():
    # Return a tensor on GPU compatible with the kernel
    return [torch.rand(batch_size, dim, dtype=torch.float32).cuda()]
