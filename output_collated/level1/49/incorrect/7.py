# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_151147/code_6.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['dim']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs Max reduction over a specific dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

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
    if 'dim' in flat_state:
        state_kwargs['dim'] = flat_state['dim']
    else:
        state_kwargs['dim'] = getattr(model, 'dim')
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

# ------------------------------------------------------------------
#  CUDA kernel – max-reduction over the last dimension (dim = 2)
#  The inner loop is unrolled by a factor of 4 to maximize throughput.
# ------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

template <int UNROLL>
__global__ void max_reduce_last_dim_kernel(
        const float* __restrict__ in,
        float* __restrict__ out,
        const int B,
        const int D1,
        const int D2)
{
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= B * D1) return;

    const int b = out_idx / D1;
    const int d1 = out_idx % D1;
    const float* __restrict__ row = in + (size_t)out_idx * D2;

    float max_val = -FLT_MAX;
    int k = 0;

    // Loop unrolling for performance
    #pragma unroll
    for (; k <= D2 - UNROLL; k += UNROLL) {
        float m = row[k];
        for (int u = 1; u < UNROLL; ++u) {
            m = fmaxf(m, row[k + u]);
        }
        max_val = fmaxf(max_val, m);
    }

    // Processing tail elements
    for (; k < D2; ++k) {
        max_val = fmaxf(max_val, row[k]);
    }

    out[out_idx] = max_val;
}

void launch_max_reduce(const torch::Tensor& input, torch::Tensor& output) {
    const int B = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);

    const int elements = B * D1;
    const int threads = 256;
    const int blocks = (elements + threads - 1) / threads;

    max_reduce_last_dim_kernel<4><<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2
    );
}
"""

# ------------------------------------------------------------------
#  C++ wrapper
# ------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void launch_max_reduce(const torch::Tensor& input, torch::Tensor& output);

torch::Tensor max_reduce(torch::Tensor input, int64_t dim) {
    auto output = torch::empty({input.size(0), input.size(1)}, input.options());
    launch_max_reduce(input, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reduce", &max_reduce, "Max reduction over last dimension");
}
"""

# Build the extension
fused_ext = load_inline(
    name="max_reduce_optimized",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
)

def functional_model(x, *, dim):
    """
    Optimized functional_model using custom CUDA kernel.
    Assumes x is a 3D float32 CUDA tensor and dim is 2.
    """
    return fused_ext.max_reduce(x, dim)

# Compatibility with the target harness
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_init_inputs():
    return [1]

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device="cuda")
    return [x]
