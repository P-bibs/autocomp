# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_185153/code_4.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_optimized_kernel(const float* __restrict__ input, 
                                     float* __restrict__ output, 
                                     size_t n) {
    // Grid-stride loop for better GPU occupancy and scalability
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Cast pointers to float4 for vectorized 128-bit memory ops
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);
    
    // Process in chunks of 4 elements (float4)
    for (size_t i = tid; i < n / 4; i += stride) {
        float4 val = input4[i];
        val.x = fmaxf(val.x, 0.0f);
        val.y = fmaxf(val.y, 0.0f);
        val.z = fmaxf(val.z, 0.0f);
        val.w = fmaxf(val.w, 0.0f);
        output4[i] = val;
    }
    
    // Handle remainder elements (less than 4) sequentially with one thread
    if (tid == 0) {
        for (size_t i = (n / 4) * 4; i < n; ++i) {
            output[i] = fmaxf(input[i], 0.0f);
        }
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    const size_t n = input.numel();
    const int threads = 256;
    // Limit blocks for better SM utilization on RTX 2080Ti
    const int blocks = min((int)((n / 4 + threads - 1) / threads), 65535);
    
    relu_optimized_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        n
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Optimized ReLU with grid-stride float4 loop");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    fused_ext.fused_op(x, output)
    return output
