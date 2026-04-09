# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_103120/code_5.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['linear_weight', 'linear_bias']
REQUIRED_FLAT_STATE_NAMES = ['linear_weight', 'linear_bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Mish, and applies Mish again.
    """

    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

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
    # State for linear (nn.Linear)
    if 'linear_weight' in flat_state:
        state_kwargs['linear_weight'] = flat_state['linear_weight']
    else:
        state_kwargs['linear_weight'] = getattr(model.linear, 'weight', None)
    if 'linear_bias' in flat_state:
        state_kwargs['linear_bias'] = flat_state['linear_bias']
    else:
        state_kwargs['linear_bias'] = getattr(model.linear, 'bias', None)
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

# Optimization: Fusing linear projection and double Mish activation into a single kernel.
# The naive implementation uses a simple dot product. For production-grade performance,
# one would typically use cuBLAS or CUTLASS, but this custom kernel demonstrates 
# the fusion concept as requested.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float mish(float x) {
    // Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    return x * tanhf(logf(1.0f + expf(x)));
}

__global__ void fused_linear_mish_kernel(const float* __restrict__ x, 
                                          const float* __restrict__ w, 
                                          const float* __restrict__ b, 
                                          float* __restrict__ out, 
                                          int batch_size, int in_features, int out_features) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < out_features) {
        float sum = b[col];
        // Compute dot product for linear layer manually
        for (int i = 0; i < in_features; ++i) {
            sum += x[row * in_features + i] * w[col * in_features + i];
        }
        
        // Fused activations in registers
        float m1 = mish(sum);
        float m2 = mish(m1);
        
        out[row * out_features + col] = m2;
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = w.size(0);

    const dim3 threads(16, 16);
    const dim3 blocks((out_features + 15) / 16, (batch_size + 15) / 16);

    fused_linear_mish_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), 
        out.data_ptr<float>(), batch_size, in_features, out_features
    );
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Linear + 2xMish Forward");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, linear_weight, linear_bias):
    """
    Computes the fused Linear Transformation followed by two sequential Mish activations.
    """
    batch_size = x.size(0)
    out_features = linear_weight.size(0)
    
    # Pre-allocate output buffer
    out = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
    
    # Run the custom fused kernel
    fused_ext.fused_op(x, linear_weight, linear_bias, out)
    
    return out

# Constants provided for testing requirements
batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda')]
