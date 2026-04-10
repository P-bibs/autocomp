# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_143434/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['dim']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs sum reduction over a specified dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
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

# -------------------------------------------------------------------------
#  CUDA source – optimized reduction kernel with shared memory reduction
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Optimized kernel using shared memory reduction
__global__ void sum_dim1_optimized_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          const int B, const int D1, const int D2) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int b = bid / D2;
    const int d = bid % D2;
    
    if (b >= B || d >= D2) return;

    // Each thread loads multiple elements if needed
    const int elements_per_thread = (D1 + blockDim.x - 1) / blockDim.x;
    float sum = 0.0f;
    
    for (int i = 0; i < elements_per_thread; ++i) {
        const int idx = tid + i * blockDim.x;
        if (idx < D1) {
            sum += input[b * D1 * D2 + idx * D2 + d];
        }
    }
    
    // Shared memory for block-level reduction
    __shared__ float sdata[1024];
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduce within block using shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[b * D2 + d] = sdata[0];
    }
}

// Host-side wrapper
void sum_dim1_optimized(torch::Tensor input, torch::Tensor output) {
    const int B = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);

    const int threads = 1024;
    const int blocks = B * D2;
    
    dim3 grid(blocks);
    dim3 block(threads);

    sum_dim1_optimized_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2);
}
"""

# -------------------------------------------------------------------------
#  C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void sum_dim1_optimized(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1_optimized", &sum_dim1_optimized,
          "Optimized sum along dimension 1 using shared memory reduction");
}
"""

# Compile the extension
sum_ext_optimized = load_inline(
    name='sum_dim1_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
#  Functional model – uses the optimized reduction kernel
# -------------------------------------------------------------------------
def functional_model(x, *, dim):
    """
    Reduce input tensor `x` of shape (B, D1, D2) along dimension 1 using a
    custom CUDA kernel optimized with shared memory reduction.
    """
    assert dim == 1, "Only reduction along dimension 1 is supported."
    
    B, D1, D2 = x.shape
    
    # Output tensor with shape (B, D2)
    output = torch.zeros((B, D2), device=x.device, dtype=x.dtype)
    
    # Launch the optimized kernel
    sum_ext_optimized.sum_dim1_optimized(x, output)
    
    # Reshape to (B, 1, D2) to match expected output format
    return output.unsqueeze(1)

# -------------------------------------------------------------------------
#  Quick sanity-check when the file is executed directly
# -------------------------------------------------------------------------
if __name__ == "__main__":
    batch_size = 128
    dim1 = 4096
    dim2 = 4095
    reduce_dim = 1

    x = torch.rand(batch_size, dim1, dim2, device='cuda', dtype=torch.float32)
    out = functional_model(x, dim=reduce_dim)

    # Shape check
    assert out.shape == (batch_size, 1, dim2), f"Wrong shape: {out.shape}"

    # Result check against PyTorch's native reduction
    ref = x.sum(dim=1, keepdim=True)
    assert torch.allclose(out, ref, atol=1e-4), "Results diverge!"
    print("✓ functional_model with optimized kernel works correctly.")
