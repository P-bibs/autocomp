# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_235928/code_22.py
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

# -------------------------------------------------------------------------
# CUDA kernel – float4 vectorization + grid-stride loop for high occupancy
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_kernel_vec4_occ(
    const float* __restrict__ input,
    float*       __restrict__ output,
    const size_t numel)
{
    // Grid-stride loop: each thread processes multiple float4 chunks.
    // This keeps the Grid small (launch overhead low) while saturating the GPU.
    size_t idx = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const size_t stride = ((size_t)blockDim.x * gridDim.x) * 4;

    for (size_t i = idx; i < numel; i += stride) {
        if (i + 3 < numel) {
            float4 in_vec = reinterpret_cast<const float4*>(input)[i >> 2];
            float4 out_vec;
            out_vec.x = tanhf(in_vec.x);
            out_vec.y = tanhf(in_vec.y);
            out_vec.z = tanhf(in_vec.z);
            out_vec.w = tanhf(in_vec.w);
            reinterpret_cast<float4*>(output)[i >> 2] = out_vec;
        } else {
            // Tail cleanup for remaining elements
            for (size_t j = i; j < numel; ++j) {
                output[j] = tanhf(input[j]);
            }
            break; 
        }
    }
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const size_t numel = input.numel();
    const int threads_per_block = 512; // Balanced occupancy for 2080Ti
    
    // Heuristic: We want enough blocks to saturate physical SMs without
    // creating thousands of unnecessary pending blocks. 
    // 128 blocks * 512 threads = 65,536 threads per wave.
    int blocks = 128; 
    
    tanh_kernel_vec4_occ<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel);
}
"""

# -------------------------------------------------------------------------
# C++ binding – exposes the kernel to Python
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output);

torch::Tensor custom_tanh(const torch::Tensor& input) {
    auto output = torch::empty_like(input);
    launch_tanh_kernel(input, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_tanh", &custom_tanh, "Vectorized CUDA tanh with grid-stride");
}
"""

# -------------------------------------------------------------------------
# Build/Compile extension
# -------------------------------------------------------------------------
tanh_ext = load_inline(
    name='tanh_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Required Interface
# -------------------------------------------------------------------------
batch_size = 4096
dim = 393216

def functional_model(x):
    return tanh_ext.custom_tanh(x)

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
