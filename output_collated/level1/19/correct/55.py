# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_190701/code_3.py
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

# --- Optimized CUDA Kernel Using Coalesced Load + Vectorization Where Safe ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_forward_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        const size_t n) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_floats = n;
    
    // Coalesced memory access: each thread reads from consecutive addresses
    if (tid * 4 + 3 < total_floats) {
        float4 val = reinterpret_cast<const float4*>(input)[tid];
        float4 result;
        result.x = fmaxf(val.x, 0.0f);
        result.y = fmaxf(val.y, 0.0f);
        result.z = fmaxf(val.z, 0.0f);
        result.w = fmaxf(val.w, 0.0f);
        reinterpret_cast<float4*>(output)[tid] = result;
    } else {
        // Tail case: process element-wise (still coalesced)
        for (int i = 0; i < 4; ++i) {
            size_t idx = tid * 4 + i;
            if (idx < total_floats) {
                float val = input[idx];
                output[idx] = fmaxf(val, 0.0f);  // ReLU
            }
        }
    }
}

void fused_op_forward(int blocks, int threads, torch::Tensor input, torch::Tensor output) {
    size_t n = input.numel();
    fused_op_forward_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(int blocks, int threads, torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Optimized Coalesced ReLU");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(x):
    output = torch.empty_like(x)
    numel = x.numel()
    threads = 256
    blocks = (numel + threads * 4 - 1) // (threads * 4)  # One thread handles 4 floats
    fused_ext.fused_op(blocks, threads, x, output)
    return output

# Evaluation setup
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]
