# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_152339/code_0.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- CUDA Kernel Code ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

template<int BLOCK_SIZE>
__global__ void max_dim2_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int64_t batch_size,
    const int64_t dim1,
    const int64_t dim2
) {
    const int64_t batch_dim1_idx = blockIdx.x;
    if (batch_dim1_idx >= batch_size * dim1) return;

    const float* row = input + batch_dim1_idx * dim2;
    float max_val = row[0];

    // Reduce within thread
    for (int64_t i = threadIdx.x; i < dim2; i += blockDim.x) {
        max_val = fmaxf(max_val, row[i]);
    }

    // Shared memory for block-wide reduction
    __shared__ float sdata[BLOCK_SIZE];
    sdata[threadIdx.x] = max_val;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (threadIdx.x == 0) {
        output[batch_dim1_idx] = sdata[0];
    }
}

void launch_max_dim2_kernel(
    const at::Tensor& input,
    at::Tensor& output,
    const int64_t batch_size,
    const int64_t dim1,
    const int64_t dim2
) {
    const int64_t total_outer_elements = batch_size * dim1;
    const int threads_per_block = 256;
    const int blocks = total_outer_elements;

    max_dim2_kernel<256><<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim1,
        dim2
    );
}
"""

# --- C++ Binding Code ---
cpp_source = r"""
#include <torch/extension.h>

void launch_max_dim2_kernel(
    const at::Tensor& input,
    at::Tensor& output,
    const int64_t batch_size,
    const int64_t dim1,
    const int64_t dim2
);

torch::Tensor fused_max_dim2(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
    
    const auto sizes = input.sizes();
    TORCH_CHECK(sizes.size() == 3, "Input tensor must be 3D");
    
    const int64_t batch_size = sizes[0];
    const int64_t dim1 = sizes[1];
    const int64_t dim2 = sizes[2];
    
    auto output = torch::empty({batch_size, dim1}, input.options());
    
    launch_max_dim2_kernel(input, output, batch_size, dim1, dim2);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_max_dim2", &fused_max_dim2, "Fused Max over last dimension");
}
"""

# --- Compile the Extension ---
max_dim2_ext = load_inline(
    name='max_dim2_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    dim,
):
    # For performance, we optimize specifically for dim=2 (last dimension)
    # This aligns with the large dimension size (4095) in get_inputs()
    if dim == 2 or dim == -1:
        return max_dim2_ext.fused_max_dim2(x.contiguous())
    else:
        # For other dimensions, fall back to PyTorch's implementation
        # In a full optimization, we'd write kernels for all dimensions
        return torch.max(x, dim=dim)[0]

batch_size = 128
dim1 = 4096
dim2 = 4095

def get_init_inputs():
    return [2]  # Optimized for reduction over last dimension

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device='cuda')
    return [x]
