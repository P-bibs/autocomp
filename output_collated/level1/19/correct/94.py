# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_193747/code_31.py
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

# Optimization: Grid-stride loop + Vectorized float4
# This pattern is robust to any tensor size and maximizes L2 cache throughput.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void fused_op_forward_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n) {
    size_t vec_n = n / 4;
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
    
    // Grid-stride loop for vectorized operations
    for (size_t i = idx; i < vec_n; i += blockDim.x * gridDim.x) {
        float4 in_vec = reinterpret_cast<const float4*>(input)[i];
        float4 out_vec;
        out_vec.x = (in_vec.x > 0.0f) ? in_vec.x : 0.0f;
        out_vec.y = (in_vec.y > 0.0f) ? in_vec.y : 0.0f;
        out_vec.z = (in_vec.z > 0.0f) ? in_vec.z : 0.0f;
        out_vec.w = (in_vec.w > 0.0f) ? in_vec.w : 0.0f;
        reinterpret_cast<float4*>(output)[i] = out_vec;
    }

    // Tail handling for dimensions not divisible by 4
    for (size_t i = idx * 4 + (vec_n * 4); i < n; i += blockDim.x * gridDim.x * 4) {
        for (int j = 0; j < 4 && (i + j) < n; ++j) {
            float val = input[i + j];
            output[i + j] = (val > 0.0f) ? val : 0.0f;
        }
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    size_t n = input.numel();
    const int threads = 256;
    // Occupancy: 128*256 fits well for memory-bound ops on 2080Ti
    const int blocks = std::min((size_t)1024, (n / 4 + threads - 1) / threads);
    
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

# Compile extension with specific architecture tags for RTX 2080Ti
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(x):
    """
    Applies optimized ReLU. 
    Maintains semantic equivalence to PyTorch ReLU.
    """
    output = torch.empty_like(x)
    fused_ext.fused_op(x, output)
    return output

# Verification/Usage
if __name__ == "__main__":
    batch_size = 4096
    dim = 393216
    x = torch.randn(batch_size, dim, device='cuda')
    y = functional_model(x)
    # Verification against native PyTorch implementation
    assert torch.allclose(y, torch.nn.functional.relu(x), atol=1e-6)
    print("Optimization test passed.")
