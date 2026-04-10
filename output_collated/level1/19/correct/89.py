# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_192155/code_29.py
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

# The CUDA kernel uses a grid-stride loop for scalability and float4 
# vectorization to maximize memory bandwidth utilization.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_forward_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // We work on blocks of 4 floats (float4) to ensure coalesced 128-bit memory access
    // Each thread processes chunks of 4 floats.
    size_t n_vec = n / 4;
    for (size_t i = tid; i < n_vec; i += stride) {
        float4 in_vec = reinterpret_cast<const float4*>(input)[i];
        float4 out_vec;
        out_vec.x = fmaxf(in_vec.x, 0.0f);
        out_vec.y = fmaxf(in_vec.y, 0.0f);
        out_vec.z = fmaxf(in_vec.z, 0.0f);
        out_vec.w = fmaxf(in_vec.w, 0.0f);
        reinterpret_cast<float4*>(output)[i] = out_vec;
    }

    // Handle trailing elements if n is not divisible by 4
    if (tid == 0) {
        for (size_t i = (n / 4) * 4; i < n; ++i) {
            output[i] = fmaxf(input[i], 0.0f);
        }
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    size_t n = input.numel();
    const int threads = 256; // Standard block size for high occupancy
    const int blocks = 1024; // Sufficient to cover SMs for 2080Ti
    fused_op_forward_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Optimized ReLU using Grid-Stride and float4");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    # Ensure memory is contiguous for coalesced access
    if not x.is_contiguous():
        x = x.contiguous()
    output = torch.empty_like(x)
    fused_ext.fused_op(x, output)
    return output

if __name__ == "__main__":
    # Setup for verification
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim, device='cuda')
    out = functional_model(x)
    assert torch.allclose(out, torch.relu(x))
    print("Optimization verified successfully.")
