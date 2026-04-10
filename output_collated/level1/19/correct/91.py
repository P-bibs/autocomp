# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_192155/code_28.py
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

# -------------------------------------------------------------------------
# CUDA source – Optimized fused ReLU kernel
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Vectorized ReLU kernel using float4
// Block size 256 is chosen to balance register pressure and occupancy
__global__ void relu_fused_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t base = tid * 4;
    
    if (base >= n) return;

    size_t remaining = n - base;

    // Bulk path: full float4 vector load/store
    if (remaining >= 4) {
        float4 in_vec = reinterpret_cast<const float4*>(input)[tid];
        in_vec.x = fmaxf(in_vec.x, 0.0f);
        in_vec.y = fmaxf(in_vec.y, 0.0f);
        in_vec.z = fmaxf(in_vec.z, 0.0f);
        in_vec.w = fmaxf(in_vec.w, 0.0f);
        reinterpret_cast<float4*>(output)[tid] = in_vec;
    } 
    // Tail path: scalar cleanup for sizes not divisible by 4
    else {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (base + i < n) {
                output[base + i] = fmaxf(input[base + i], 0.0f);
            }
        }
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    size_t n = input.numel();
    constexpr int Threads = 256;
    size_t vector_cnt = (n + 3) / 4;
    int blocks = (int)((vector_cnt + Threads - 1) / Threads);

    relu_fused_kernel<<<blocks, Threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
}
"""

# -------------------------------------------------------------------------
# C++ binding
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused ReLU kernel with float4 vectorization");
}
"""

# -------------------------------------------------------------------------
# Build the inline extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional wrapper
# -------------------------------------------------------------------------
def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized fused ReLU using float4 vectorization and 256-thread blocks.
    Ensures contiguous memory access for coalescing.
    """
    # Enforce contiguous layout for float4 vectorized loading
    if not x.is_contiguous():
        x = x.contiguous()
        
    output = torch.empty_like(x)
    fused_ext.fused_op(x, output)
    return output
