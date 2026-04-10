# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_134650/code_31.py
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

# -----------------------------------------------------------------------------
# CUDA Kernel: Sum along dim=1 with coalesced memory access and loop unrolling.
# 
# Performance optimizations:
# 1. Coalesced Reads: Each thread processes 4 consecutive floats in the D2 
#    dimension, ensuring threads in a warp access contiguous memory.
# 2. Loop Unrolling: #pragma unroll 8 reduces the loop control overhead 
#    and allows the compiler to pipeline arithmetic effectively.
# 3. Register Accumulation: Using registers (sum0..sum3) instead of local 
#    arrays avoids stack spill and increases ILP.
# -----------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_dim1_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int B, int D1, int D2) {
    int b = blockIdx.x;
    int col_idx = (blockIdx.y * blockDim.x + threadIdx.x) * 4;

    if (b < B && col_idx < D2) {
        // Calculate bounds for safety
        int remaining = D2 - col_idx;
        int vec_len = (remaining < 4) ? remaining : 4;

        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        
        // Offset into the flattened input
        // input shape is (B, D1, D2)
        const float* input_ptr = input + (b * D1 * D2) + col_idx;

        #pragma unroll 8
        for (int i = 0; i < D1; ++i) {
            int row_offset = i * D2;
            if (vec_len > 0) sum0 += input_ptr[row_offset];
            if (vec_len > 1) sum1 += input_ptr[row_offset + 1];
            if (vec_len > 2) sum2 += input_ptr[row_offset + 2];
            if (vec_len > 3) sum3 += input_ptr[row_offset + 3];
        }

        // Store result back to output (B, D2)
        float* out_ptr = output + (b * D2) + col_idx;
        if (vec_len > 0) out_ptr[0] = sum0;
        if (vec_len > 1) out_ptr[1] = sum1;
        if (vec_len > 2) out_ptr[2] = sum2;
        if (vec_len > 3) out_ptr[3] = sum3;
    }
}

void sum_dim1(torch::Tensor input, torch::Tensor output) {
    const int B  = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);

    // Threads per block configuration
    const int threads_per_block = 256;
    // Each thread processes 4 elements width-wise
    int blocks_y = (D2 / 4 + threads_per_block - 1) / threads_per_block;
    
    dim3 threads(threads_per_block);
    dim3 blocks(B, blocks_y);

    sum_dim1_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        B, D1, D2
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void sum_dim1(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1", &sum_dim1, "Sum along dim 1 with unrolled kernel");
}
"""

# Compile the extension
sum_ext = load_inline(
    name='sum_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    """
    Optimized sum along dim 1.
    """
    assert dim == 1
    # Output shape should be (B, 1, D2) as per original requirements
    batch_size, _, d2 = x.shape
    output = torch.empty((batch_size, d2), device=x.device, dtype=x.dtype)
    
    sum_ext.sum_dim1(x, output)
    
    # Return with original dimension structure
    return output.unsqueeze(1)
