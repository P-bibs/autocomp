# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_192155/code_15.py
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

# ----------------------------------------------------------------------
# Optimised CUDA kernel – grid‑stride vectorised ReLU
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_forward_kernel(const float* __restrict__ input,
                                         float*       __restrict__ output,
                                         const size_t n)
{
    // Grid‑stride: each thread processes many float4 chunks.
    const size_t stride = blockDim.x * gridDim.x * 4;      // elements per thread per iteration
    const size_t idx    = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    for (size_t i = idx; i < n; i += stride) {
        // ----- vectorised path (float4) -----
        if (i + 3 < n) {
            float4 in_vec = reinterpret_cast<const float4*>(input)[i / 4];
            float4 out_vec;
            // fmaxf is a single‑instruction fast ReLU
            out_vec.x = fmaxf(in_vec.x, 0.0f);
            out_vec.y = fmaxf(in_vec.y, 0.0f);
            out_vec.z = fmaxf(in_vec.z, 0.0f);
            out_vec.w = fmaxf(in_vec.w, 0.0f);
            reinterpret_cast<float4*>(output)[i / 4] = out_vec;
        }
        else {
            // ----- tail (≤3 elements) – fully unrolled -----
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (i + j < n) {
                    float val = input[i + j];
                    output[i + j] = fmaxf(val, 0.0f);
                }
            }
        }
    }
}

// Host‑side wrapper: choose a modest number of blocks to keep launch overhead low
void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    const size_t n          = input.numel();
    const int    threads    = 256;           // block size
    const int    max_blocks = 8192;          // cap the grid

    // Original launch would be (n + 1023) / 1024 blocks – far too many.
    // Now we cap the grid; each thread will loop over the whole data.
    int blocks = (n + threads * 4 - 1) / (threads * 4);
    if (blocks > max_blocks) blocks = max_blocks;

    fused_op_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);

    // Ensure the kernel finishes before the wrapper returns
    cudaDeviceSynchronize();
}
"""

# ----------------------------------------------------------------------
# C++ binding (PYBIND11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward,
          "Optimized vectorized ReLU with grid‑stride loop");
}
"""

# ----------------------------------------------------------------------
# Build the inline extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional model used by the evaluator
# ----------------------------------------------------------------------
def functional_model(x: torch.Tensor) -> torch.Tensor:
    """Apply the optimized fused ReLU."""
    output = torch.empty_like(x)
    fused_ext.fused_op(x, output)
    return output

# ----------------------------------------------------------------------
# Evaluation helpers (not required to modify)
# ----------------------------------------------------------------------
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]
