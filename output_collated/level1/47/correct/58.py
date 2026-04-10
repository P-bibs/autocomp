# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_131936/code_31.py
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
# CUDA Kernel Optimization Strategy:
# 1. Coalesced Memory Access: By assigning each thread to a specific column output 'j',
#    consecutive threads read consecutive memory addresses across the inner loop (i),
#    maximizing the use of the L1/L2 cache and memory bandwidth.
# 2. Register Pressure Mitigation: Using a single float accumulator per thread 
#    prevents register spilling, allowing for higher occupancy on the SM.
# 3. Instruction Throughput: The #pragma unroll directive minimizes branch overhead, 
#    and the grid configuration ensures efficient distribution across Streaming Multiprocessors.
# -----------------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_dim1_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int B, int D1, int D2) {
    // Map thread index to (b, j) coordinate
    int b = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;

    if (b < B && j < D2) {
        float sum = 0.0f;
        const float* input_ptr = input + b * D1 * D2 + j;

        // Process along dimension 1. Loop unrolling helps hiding latency.
        #pragma unroll 8
        for (int i = 0; i < D1; ++i) {
            sum += input_ptr[i * D2];
        }

        output[b * D2 + j] = sum;
    }
}

void sum_dim1(torch::Tensor input, torch::Tensor output) {
    int B = input.size(0);
    int D1 = input.size(1);
    int D2 = input.size(2);

    // Use a warp-friendly block size
    const int threads_x = 256;
    // Grid sizing: X dimension for Batch, Y dimension for columns
    dim3 threads(threads_x);
    dim3 blocks(B, (D2 + threads_x - 1) / threads_x);

    sum_dim1_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void sum_dim1(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1", &sum_dim1, "Sum along dim=1 with coalesced memory access");
}
"""

# Compile the extension
sum_ext = load_inline(
    name='sum_dim1_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    """
    Optimized sum along dimension 1 using a custom managed CUDA kernel.
    Input: (B, D1, D2) -> Output: (B, 1, D2)
    """
    assert dim == 1, "Only dim=1 is supported"
    
    # Pre-allocate output tensor
    output = torch.empty((x.shape[0], x.shape[2]), device=x.device, dtype=x.dtype)
    
    # Call to compiled CUDA kernel
    sum_ext.sum_dim1(x, output)
    
    # Match the expected output shape (B, 1, D2)
    return output.unsqueeze(1)

# --- Test/Evaluation Parameters ---
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_init_inputs():
    return [1]

def get_inputs():
    return [torch.rand(batch_size, dim1, dim2, device='cuda')]
