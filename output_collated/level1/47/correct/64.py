# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_134650/code_13.py
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
#  CUDA source – vectorized reduction kernel with float4 loads and loop unrolling
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Vectorized kernel using float4 loads for better memory coalescing
__global__ void sum_dim1_float4_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       const int B, const int D1, const int D2) {
    const int tid = threadIdx.x;
    const int b = blockIdx.x;
    const int elements_per_thread = 4;
    const int elements_per_block = blockDim.x * elements_per_thread;
    const int j_base = blockIdx.y * elements_per_block + tid * elements_per_thread;

    if (b >= B) return;

    // Handle cases where D2 is not a multiple of 4
    const bool aligned = (D2 % 4 == 0) && (j_base + 3 < D2);
    const int remaining = D2 - j_base;

    if (aligned) {
        // Use float4 for aligned memory access
        float4 sum_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        
        #pragma unroll 8
        for (int i = 0; i < D1; ++i) {
            const float4 val = __ldg((const float4*)&input[b * D1 * D2 + i * D2 + j_base]);
            sum_vec.x += val.x;
            sum_vec.y += val.y;
            sum_vec.z += val.z;
            sum_vec.w += val.w;
        }
        
        output[b * D2 + j_base + 0] = sum_vec.x;
        output[b * D2 + j_base + 1] = sum_vec.y;
        output[b * D2 + j_base + 2] = sum_vec.z;
        output[b * D2 + j_base + 3] = sum_vec.w;
    } else {
        // Fallback to scalar loads for unaligned or boundary cases
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        
        #pragma unroll 8
        for (int i = 0; i < D1; ++i) {
            const int offset = b * D1 * D2 + i * D2 + j_base;
            if (j_base + 0 < D2) sum0 += __ldg(&input[offset + 0]);
            if (j_base + 1 < D2) sum1 += __ldg(&input[offset + 1]);
            if (j_base + 2 < D2) sum2 += __ldg(&input[offset + 2]);
            if (j_base + 3 < D2) sum3 += __ldg(&input[offset + 3]);
        }
        
        if (j_base + 0 < D2) output[b * D2 + j_base + 0] = sum0;
        if (j_base + 1 < D2) output[b * D2 + j_base + 1] = sum1;
        if (j_base + 2 < D2) output[b * D2 + j_base + 2] = sum2;
        if (j_base + 3 < D2) output[b * D2 + j_base + 3] = sum3;
    }
}

// Host-side wrapper
void sum_dim1_float4(torch::Tensor input, torch::Tensor output) {
    const int B = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);

    const int threads = 256;
    const int elements_per_thread = 4;
    const int elements_per_block = threads * elements_per_thread;
    
    const int blocks_x = B;
    const int blocks_y = (D2 + elements_per_block - 1) / elements_per_block;
    
    dim3 grid(blocks_x, blocks_y);
    dim3 block(threads);

    sum_dim1_float4_kernel<<<grid, block>>>(
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

void sum_dim1_float4(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1_float4", &sum_dim1_float4,
          "Optimized sum along dimension 1 using float4 memory loads");
}
"""

# Compile the extension
sum_ext_float4 = load_inline(
    name='sum_dim1_float4',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
#  Functional model – uses the custom float4-optimized kernel
# -------------------------------------------------------------------------
def functional_model(x, *, dim):
    """
    Reduce input tensor `x` of shape (B, D1, D2) along dimension 1 using a
    custom CUDA kernel optimized with float4 memory loads and loop unrolling.
    """
    assert dim == 1, "Only reduction along dimension 1 is supported."
    
    B, D1, D2 = x.shape
    
    # Output tensor with shape (B, D2)
    output = torch.zeros((B, D2), device=x.device, dtype=x.dtype)
    
    # Launch the optimized kernel
    sum_ext_float4.sum_dim1_float4(x, output)
    
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
    print("✓ functional_model with float4-optimized kernel works correctly.")
