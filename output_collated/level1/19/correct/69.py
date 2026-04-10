# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_190701/code_14.py
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
# CUDA kernel: each thread processes 8 float4 vectors (32 elements)
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_forward_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        size_t n) {
    // 8 float4 vectors per thread → 32 elements
    constexpr int elements_per_thread = 8;   // 8 * 4 = 32 elements per thread

    // Total number of threads in the grid
    size_t total_threads = blockDim.x * gridDim.x;
    // Stride = total work done by one thread across the whole input
    size_t stride = total_threads * elements_per_thread * 4;

    // Starting index for this thread
    size_t start = (blockIdx.x * blockDim.x + threadIdx.x) * elements_per_thread * 4;

    // Loop over the data with the computed stride
    for (size_t idx = start; idx < n; idx += stride) {
        // Process each float4 vector in the inner loop
        for (int i = 0; i < elements_per_thread; ++i) {
            size_t i_idx = idx + i * 4;
            if (i_idx + 3 < n) {
                // Vectorized load through the L1/texture cache
                float4 in_vec = __ldg(reinterpret_cast<const float4*>(input) + i_idx / 4);
                // ReLU
                float4 out_vec;
                out_vec.x = fmaxf(in_vec.x, 0.0f);
                out_vec.y = fmaxf(in_vec.y, 0.0f);
                out_vec.z = fmaxf(in_vec.z, 0.0f);
                out_vec.w = fmaxf(in_vec.w, 0.0f);
                // Coalesced vectorized store
                reinterpret_cast<float4*>(output)[i_idx / 4] = out_vec;
            } else {
                // Tail: handle up to three remaining elements with scalar loads
                for (int j = 0; j < 4 && i_idx + j < n; ++j) {
                    float val = __ldg(input + i_idx + j);
                    output[i_idx + j] = fmaxf(val, 0.0f);
                }
            }
        }
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    size_t n = input.numel();
    const int threads = 256;                // block size
    constexpr int elements_per_thread = 8;  // must match kernel constant

    // Compute grid size based on the increased work per thread
    const int blocks = (n + threads * elements_per_thread * 4 - 1) /
                       (threads * elements_per_thread * 4);

    fused_op_forward_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                                output.data_ptr<float>(),
                                                n);
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward,
          "ReLU kernel with increased per‑thread work");
}
"""

# -------------------------------------------------------------------------
# Build the inline extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional model to be imported for evaluation
# -------------------------------------------------------------------------
def functional_model(x: torch.Tensor) -> torch.Tensor:
    """In‑place ReLU implemented as a custom CUDA kernel with higher per‑thread work."""
    output = torch.empty_like(x)
    fused_ext.fused_op(x, output)
    return output


# -------------------------------------------------------------------------
# Evaluation‑setup helpers (unused by the grader)
# -------------------------------------------------------------------------
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]
