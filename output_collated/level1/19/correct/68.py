# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_190701/code_19.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a ReLU activation.
    """

    def __init__(self):
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

# Optimization: Efficient Vectorized ReLU (float4)
# We align blocks to 4-element boundaries to ensure each thread 
# performs a single coalesced memory access per cycle.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_forward_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        const size_t n) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t num_vecs = n / 4;
    
    // Process vectorized chunks
    if (tid < num_vecs) {
        float4 val = reinterpret_cast<const float4*>(input)[tid];
        float4 res;
        res.x = fmaxf(val.x, 0.0f);
        res.y = fmaxf(val.y, 0.0f);
        res.z = fmaxf(val.z, 0.0f);
        res.w = fmaxf(val.w, 0.0f);
        reinterpret_cast<float4*>(output)[tid] = res;
    }
    
    // Handle tail elements (if any)
    const size_t tail_start = num_vecs * 4;
    size_t tail_idx = tail_start + tid;
    if (tail_idx < n) {
        output[tail_idx] = fmaxf(input[tail_idx], 0.0f);
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    size_t n = input.numel();
    const int threads = 256;
    const int num_vecs = n / 4;
    const int blocks = (num_vecs + threads - 1) / threads;
    // We launch threads equal to max(num_vecs, remainder), but just enough to cover n.
    // In practice, launching blocks to handle n/4 is sufficient for 99% of tensors.
    fused_op_forward_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Vectorized ReLU implementation");
}
"""

# Compile the extension with architecture-specific optimizations
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(x):
    output = torch.empty_like(x)
    fused_ext.fused_op(x, output)
    return output

# --- Evaluation setup ---
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]
