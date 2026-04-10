# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_190701/code_16.py
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
# Optimized CUDA kernel: 
# Uses grid-stripping to process multiple 8-float chunks per thread, 
# significantly improving instruction-level parallelism.
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_fused_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop: each thread processes multiple chunks of 8 floats
    // This hides latency and keeps the GPU busy across different tensor sizes
    for (size_t i = tid * 8; i < n; i += stride * 8) {
        if (i + 7 < n) {
            // Bulk path: Process 8 elements at once using two float4s
            float4 in1 = __ldg(reinterpret_cast<const float4*>(input + i));
            float4 in2 = __ldg(reinterpret_cast<const float4*>(input + i + 4));
            
            float4 out1, out2;
            out1.x = fmaxf(in1.x, 0.0f); out1.y = fmaxf(in1.y, 0.0f);
            out1.z = fmaxf(in1.z, 0.0f); out1.w = fmaxf(in1.w, 0.0f);
            out2.x = fmaxf(in2.x, 0.0f); out2.y = fmaxf(in2.y, 0.0f);
            out2.z = fmaxf(in2.z, 0.0f); out2.w = fmaxf(in2.w, 0.0f);
            
            reinterpret_cast<float4*>(output + i)[0] = out1;
            reinterpret_cast<float4*>(output + i + 4)[0] = out2;
        } else {
            // Remainder path: Unrolled handling for alignment edges
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (i + j < n) {
                    float val = __ldg(input + i + j);
                    output[i + j] = fmaxf(val, 0.0f);
                }
            }
        }
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    size_t n = input.numel();
    int threads = 256; // Standard block size for occupancy
    int blocks = (n + (8 * threads) - 1) / (8 * threads);
    blocks = (blocks > 65535) ? 65535 : blocks; // Clamp for safety
    
    relu_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused ReLU kernel");
}
"""

# Compile extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    """Applies ReLU on `x` using a grid-strided fused kernel."""
    output = torch.empty_like(x)
    fused_ext.fused_op(x, output)
    return output

# Benchmark setup
batch_size = 4096
dim = 393216

def get_init_inputs(): return []
def get_inputs(): return [torch.rand(batch_size, dim, device='cuda')]
