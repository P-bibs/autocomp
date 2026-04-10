# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_004356/code_23.py
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

# Optimized CUDA kernel with increased ILP and explicit vectorization
# Processing 16 floats per thread (4 * float4) keeps the pipeline saturated
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_optimized_kernel(const float* __restrict__ x, float* __restrict__ out, size_t n) {
    size_t idx = (size_t)(blockIdx.x * blockDim.x + threadIdx.x) * 16;
    
    // Bounds check enforced at grid level via loop handling or masking
    // Given performance needs, we assume n is sufficiently divisible by 16 or 
    // handle remainder safely with explicit scalar cleanup to avoid bank conflicts.
    if (idx + 15 < n) {
        float4 v1 = reinterpret_cast<const float4*>(x + idx)[0];
        float4 v2 = reinterpret_cast<const float4*>(x + idx + 4)[0];
        float4 v3 = reinterpret_cast<const float4*>(x + idx + 8)[0];
        float4 v4 = reinterpret_cast<const float4*>(x + idx + 12)[0];
        
        #pragma unroll
        for(int k=0; k<4; ++k) {
            (&v1.x)[k] = tanhf((&v1.x)[k]);
            (&v2.x)[k] = tanhf((&v2.x)[k]);
            (&v3.x)[k] = tanhf((&v3.x)[k]);
            (&v4.x)[k] = tanhf((&v4.x)[k]);
        }
        
        reinterpret_cast<float4*>(out + idx)[0] = v1;
        reinterpret_cast<float4*>(out + idx + 4)[0] = v2;
        reinterpret_cast<float4*>(out + idx + 8)[0] = v3;
        reinterpret_cast<float4*>(out + idx + 12)[0] = v4;
    } else {
        for (size_t i = idx; i < n; ++i) {
            out[i] = tanhf(x[i]);
        }
    }
}

void fused_tanh_forward(torch::Tensor x, torch::Tensor out) {
    const size_t n = x.numel();
    const int threads = 128; // Lower thread count to allow more registers per thread
    const int work_per_thread = 16;
    const int blocks = (n + (threads * work_per_thread) - 1) / (threads * work_per_thread);
    
    tanh_optimized_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_tanh_forward(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh", &fused_tanh_forward, "Highly optimized Tanh forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_tanh_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(x):
    # Ensure input is contiguous; kernels using float4/vectorized access require strict alignment
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    fused_ext.fused_tanh(x, out)
    return out
