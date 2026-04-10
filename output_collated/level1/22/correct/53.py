# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_235928/code_17.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a Tanh activation.
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_vec4_kernel(const float* __restrict__ x, float* __restrict__ out, size_t n) {
    size_t idx = (size_t)(blockIdx.x * blockDim.x + threadIdx.x) * 4;
    size_t stride = (size_t)blockDim.x * gridDim.x * 4;
    
    // Process 4 elements per thread using vectorized loads/stores
    for (size_t i = idx; i < n; i += stride) {
        if (i + 3 < n) {
            float4 vec_x = reinterpret_cast<const float4*>(x + i)[0];
            float4 vec_out;
            vec_out.x = tanhf(vec_x.x);
            vec_out.y = tanhf(vec_x.y);
            vec_out.z = tanhf(vec_x.z);
            vec_out.w = tanhf(vec_x.w);
            reinterpret_cast<float4*>(out + i)[0] = vec_out;
        } else {
            // Remainder handling
            for (size_t j = i; j < n; ++j) {
                out[j] = tanhf(x[j]);
            }
        }
    }
}

void fused_tanh_forward(torch::Tensor x, torch::Tensor out) {
    const size_t n = x.numel();
    const int threads = 256;
    // 2080Ti has 68 SMs. 68 * 8 = 544 blocks provides significant headroom to hide latency.
    const int blocks = 544; 
    
    tanh_vec4_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_tanh_forward(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh", &fused_tanh_forward, "Vectorized Grid-Stride Tanh forward");
}
"""

fused_ext = load_inline(
    name='fused_tanh',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    Computes tanh(x) using a vectorized, grid-stride CUDA kernel.
    Optimized for memory bandwidth saturation and RTX 2080Ti occupancy.
    """
    out = torch.empty_like(x)
    # Ensure input is contiguous for float4 casting
    if not x.is_contiguous():
        x = x.contiguous()
    fused_ext.fused_tanh(x, out)
    return out
