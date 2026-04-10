# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_190701/code_6.py
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

__global__ void fused_op_forward_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Grid-stride loop with vectorized processing of 4 elements per iteration
    for (size_t i = tid * 4; i < n; i += stride * 4) {
        if (i + 3 < n) {
            // Load and process a float4 vector
            float4 in_vec = *reinterpret_cast<const float4*>(input + i);
            float4 out_vec;
            
            // Use #pragma unroll to maximize throughput for ReLU operations
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float val;
                if (k == 0) val = in_vec.x;
                else if (k == 1) val = in_vec.y;
                else if (k == 2) val = in_vec.z;
                else val = in_vec.w;
                
                val = fmaxf(val, 0.0f);
                
                if (k == 0) out_vec.x = val;
                else if (k == 1) out_vec.y = val;
                else if (k == 2) out_vec.z = val;
                else out_vec.w = val;
            }
            
            *reinterpret_cast<float4*>(output + i) = out_vec;
        } else {
            // Handle tail elements with scalar loads
            for (int j = 0; j < 4 && i + j < n; ++j) {
                output[i + j] = fmaxf(input[i + j], 0.0f);
            }
        }
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    size_t n = input.numel();
    const int threads = 256;
    // Optimize for occupancy on RTX 2080Ti
    const int blocks = 1024; 
    fused_op_forward_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                                output.data_ptr<float>(),
                                                n);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Optimized ReLU with Grid-Stride and Loop Unrolling");
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
