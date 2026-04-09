# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_103120/code_4.py
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

# Combined CUDA Kernel for the double Mish activation (Point-wise)
# We perform the Linear layer via cuBLAS and fuse the Mish activations
# into one element-wise kernel to maximize memory bandwidth usage.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void double_mish_kernel(float* data, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float x = data[idx];
        // Mish 1: x * tanh(softplus(x))
        float s1 = logf(1.0f + expf(x));
        float m1 = x * tanhf(s1);
        // Mish 2: m1 * tanh(softplus(m1))
        float s2 = logf(1.0f + expf(m1));
        data[idx] = m1 * tanhf(s2);
    }
}

void launch_double_mish(torch::Tensor output) {
    int num_elements = output.numel();
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    double_mish_kernel<<<blocks, threads>>>(output.data_ptr<float>(), num_elements);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_double_mish(torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_double_mish", &launch_double_mish, "Double Mish activation kernel");
}
"""

module = load_inline(
    name='fused_mish_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, linear_weight, linear_bias):
    # Perform standard optimized GEMM (x @ W.T + b)
    # Using torch.mm/addmm is a wrapper for cuBLAS, which is the 
    # most efficient way to compute linear layers on NVIDIA GPUs.
    # Linear weight is usually (out, in), so we transpose for matmul.
    output = torch.addmm(linear_bias, x, linear_weight.t())
    
    # Run custom fused point-wise kernel for the two Mish activations
    module.launch_double_mish(output)
    
    return output

# Inputs as defined in original
batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]
