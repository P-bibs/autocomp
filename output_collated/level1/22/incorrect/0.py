# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_231959/code_23.py
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

# ----------------------------------------------------------------------
# CUDA kernel (inline)
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Grid‑stride loop kernel for element‑wise tanh
// Uses __tanhf intrinsic for hardware-accelerated math
__global__ void tanh_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int64_t N) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    
    for (int64_t i = idx; i < N; i += stride) {
        output[i] = __tanhf(input[i]);
    }
}

void tanh_cuda(at::Tensor& input) {
    int64_t N = input.numel();
    float* data_ptr = input.data_ptr<float>();

    // Determine grid/block dimensions
    // Maximize occupancy for large N
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads > 32768 ? 32768 : (N + threads - 1) / threads;
    
    tanh_kernel<<<blocks, threads>>>(data_ptr, data_ptr, N);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
"""

# ----------------------------------------------------------------------
# C++ binding (PyBind11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void tanh_cuda(at::Tensor& input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tanh_cuda", &tanh_cuda, "In-place CUDA tanh kernel");
}
"""

# ----------------------------------------------------------------------
# Compile the extension logic
# ----------------------------------------------------------------------
_tanh_ext = None

def _get_ext():
    global _tanh_ext
    if _tanh_ext is None:
        _tanh_ext = load_inline(
            name='tanh_cuda_ext',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            with_cuda=True
        )
    return _tanh_ext

# ----------------------------------------------------------------------
# Optimized functional_model
# ----------------------------------------------------------------------
def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Computes tanh in-place on the GPU using a custom CUDA kernel.
    """
    # 1. Prepare tensor (GPU)
    # Ensure tensor is on GPU, contiguous, and float32
    if not x.is_cuda:
        x = x.cuda()
    
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
        
    if not x.is_contiguous():
        x = x.contiguous()

    # 2. Invoke GPU kernel
    ext = _get_ext()
    ext.tanh_cuda(x)

    # 3. Return to CPU to satisfy semantic requirements
    return x.cpu()

def get_init_inputs():
    return []

def get_inputs():
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim)
    return [x]
