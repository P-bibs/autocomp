# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_184018/code_11.py
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
# CUDA kernel – vectorized ReLU (float4)
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel_vec(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int64_t num_elements)
{
    const int64_t total_threads = blockDim.x * gridDim.x;   // total number of threads
    const int vec_size = 4;                                  // vector width

    // each thread works on a contiguous block of `vec_size` elements
    const int64_t base = (blockIdx.x * blockDim.x + threadIdx.x) * vec_size;

    // main vectorised loop
    for (int64_t i = base; i < num_elements; i += total_threads * vec_size) {
        // full vector (4 elements) fits completely
        if (i + vec_size <= num_elements) {
            float4 in = reinterpret_cast<const float4*>(input + i)[0];
            float4 out;
            out.x = fmaxf(0.0f, in.x);
            out.y = fmaxf(0.0f, in.y);
            out.z = fmaxf(0.0f, in.z);
            out.w = fmaxf(0.0f, in.w);
            reinterpret_cast<float4*>(output + i)[0] = out;
        } else {
            // handle the tail (1‑3 elements) with scalar loads
            for (int64_t j = i; j < num_elements; ++j) {
                float val = input[j];
                output[j] = fmaxf(0.0f, val);
            }
            break; // all remaining work for this thread is done
        }
    }
}

// Host‑side launcher
void relu_forward_cuda_vec(const torch::Tensor& input, torch::Tensor& output) {
    const int64_t num_elements = input.numel();
    const int threads = 512;                      // threads per block
    const int vec_size = 4;                       // vector width

    // Compute required number of blocks for the vectorised kernel
    const int64_t total_threads = threads;        // one block contains `threads` threads
    const int64_t num_threads_total = total_threads; // not used directly, see grid computation below
    const int64_t grid = (num_elements + threads * vec_size - 1) / (threads * vec_size);
    const int blocks = static_cast<int>(std::min<int64_t>(grid, 65535));

    relu_kernel_vec<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements
    );
}
"""

# -------------------------------------------------------------------------
# C++ bindings (PyBind11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void relu_forward_cuda_vec(const torch::Tensor& input, torch::Tensor& output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_relu_vec", &relu_forward_cuda_vec,
          "Vectorised ReLU forward (CUDA)");
}
"""

# -------------------------------------------------------------------------
# Build the inline extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_relu_vec_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional wrapper used by the benchmark harness
# -------------------------------------------------------------------------
def functional_model(x: torch.Tensor) -> torch.Tensor:
    # Move input to GPU if not already there
    if not x.is_cuda:
        x = x.to('cuda', non_blocking=True)

    # Pre‑allocate output on the same device
    output = torch.empty_like(x)

    # Call the vectorised CUDA ReLU kernel
    fused_ext.fused_relu_vec(x, output)

    return output


# -------------------------------------------------------------------------
# Benchmark boilerplate (kept identical to the original)
# -------------------------------------------------------------------------
batch_size = 4096
dim = 393216


def get_init_inputs():
    """No persistent state needed."""
    return []


def get_inputs():
    """
    Returns a single input tensor.
    The original code created the tensor on CPU; the functional wrapper moves it to GPU.
    """
    x = torch.rand(batch_size, dim)
    return [x]
