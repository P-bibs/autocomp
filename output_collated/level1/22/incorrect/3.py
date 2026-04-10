# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_004356/code_9.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void tanh_optimized_kernel(const float* __restrict__ x, float* __restrict__ out, size_t n) {
    // Each thread handles 4 elements (float4 vectorized)
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t vec_idx = tid * 4;
    
    // Main vectorized path for complete float4 chunks
    if (vec_idx + 3 < n) {
        float4 vec_x = reinterpret_cast<const float4*>(x)[tid];
        float4 vec_out;
        vec_out.x = tanhf(vec_x.x);
        vec_out.y = tanhf(vec_x.y);
        vec_out.z = tanhf(vec_x.z);
        vec_out.w = tanhf(vec_x.w);
        reinterpret_cast<float4*>(out)[tid] = vec_out;
    }
    // Handle remainder: distribute remainder elements across threads
    else if (vec_idx < n) {
        // This thread handles its remaining 1-3 elements
        for (size_t i = 0; i < 4 && vec_idx + i < n; ++i) {
            out[vec_idx + i] = tanhf(x[vec_idx + i]);
        }
    }
}

void fused_tanh_forward(torch::Tensor x, torch::Tensor out) {
    const size_t n = x.numel();
    const int threads = 512;  // Increased from 256 for better occupancy
    
    // Each thread handles 4 elements, so we need (n + 4*threads - 1) / (4*threads) blocks
    // But we calculate based on vec iterations: (n/4 + threads - 1) / threads
    // For large n, this simplifies to approximately n / (4 * threads)
    int blocks = (n + threads * 4 - 1) / (threads * 4);
    
    // Cap blocks to reasonable number based on RTX 2080Ti (68 SMs)
    // Aim for ~2-3 waves of work per SM for good occupancy
    const int max_blocks = 68 * 2;  // Conservative: 2 waves per SM
    if (blocks > max_blocks) {
        blocks = max_blocks;
    }
    
    tanh_optimized_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_tanh_forward(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh", &fused_tanh_forward, "Optimized Vectorized Tanh forward with maximized occupancy");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_tanh_opt_v2',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    Optimized tanh operation with maximized GPU occupancy.
    - Increases thread count per block to 512 for better occupancy
    - Distributes remainder handling across threads to eliminate divergence
    - Ensures input is contiguous for safe float4 vectorization
    """
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    fused_ext.fused_tanh(x, out)
    return out

def get_inputs():
    # Standard input generation
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
