# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_145843/code_6.py
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
# CUDA kernel: Optimized max-reduction using registers
# ------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

template <typename scalar_t>
__global__ void max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer,
    const int64_t reduction,
    const int64_t inner) {

    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t out_size = outer * inner;
    
    if (tid >= out_size) return;

    const int64_t outer_idx = tid / inner;
    const int64_t inner_idx = tid % inner;
    
    // Calculate start offset: jump blocks of size (reduction * inner)
    const int64_t base_offset = (outer_idx * reduction * inner) + inner_idx;

    scalar_t max_val = input[base_offset];
    
    // Performance optimization: unroll reduction loop to help register pressure
    #pragma unroll 8
    for (int64_t r = 1; r < reduction; ++r) {
        scalar_t val = input[base_offset + r * inner];
        if (val > max_val) {
            max_val = val;
        }
    }
    output[tid] = max_val;
}

void launch_max_reduce(const torch::Tensor& input, torch::Tensor& output, int64_t outer, int64_t reduction, int64_t inner) {
    const int threads = 256;
    const int64_t out_size = outer * inner;
    const int blocks = (out_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_kernel", ([&] {
        max_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer,
            reduction,
            inner
        );
    }));
}
"""

# ------------------------------------------------------------------
# C++ Bindings
# ------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void launch_max_reduce(const torch::Tensor& input, torch::Tensor& output, int64_t outer, int64_t reduction, int64_t inner);

torch::Tensor max_reduce_wrapper(const torch::Tensor& input, int64_t dim) {
    auto ndims = input.dim();
    if (dim < 0) dim += ndims;
    
    int64_t outer = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= input.size(i);
    int64_t reduction = input.size(dim);
    int64_t inner = 1;
    for (int64_t i = dim + 1; i < ndims; ++i) inner *= input.size(i);

    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndims; ++i) {
        if (i != dim) out_shape.push_back(input.size(i));
    }
    
    auto output = torch::empty(out_shape, input.options());
    launch_max_reduce(input, output, outer, reduction, inner);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reduce", &max_reduce_wrapper, "Fast CUDA max reduction");
}
"""

# Compile extension
fused_ext = load_inline(
    name='max_reduce_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    # Ensure memory is contiguous for kernel performance
    if not x.is_contiguous():
        x = x.contiguous()
    return fused_ext.max_reduce(x, dim)

# Unchanged required exports for harness
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device="cuda")
    return [x]
