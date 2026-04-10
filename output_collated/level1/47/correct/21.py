# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_123707/code_17.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Optimization: Each block processes a specific (b, output_segment).
// We ensure memory access is coalesced by having threads in a warp
// process adjacent outputs, and loop through dim1 contiguously.
__global__ void sum_dim1_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                int B, int D1, int D2) {
    int b = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;

    if (b < B && j < D2) {
        float sum = 0.0f;
        // Pointer to the first element for this specific j index in batch b
        const float* input_ptr = input + b * D1 * D2 + j;
        
        // Unroll loop for performance
        #pragma unroll 8
        for (int i = 0; i < D1; ++i) {
            // Strided access is unavoidable on input due to memory layout 
            // (B, D1, D2). We optimize by ensuring threadIdx.x accesses 
            // adjacent D2 indices (coalesced).
            sum += input_ptr[i * D2];
        }
        output[b * D2 + j] = sum;
    }
}

void sum_dim1(torch::Tensor input, torch::Tensor output) {
    const int B = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);
    
    // Threads per block
    int threads_per_block = 256;
    // Blocks for D2 dimension
    int num_blocks_d2 = (D2 + threads_per_block - 1) / threads_per_block;
    
    dim3 threads(threads_per_block);
    dim3 blocks(B, num_blocks_d2);
    
    sum_dim1_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), B, D1, D2
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void sum_dim1(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1", &sum_dim1, "Sum along dimension 1 with coalesced memory access");
}
"""

# Compile the extension
sum_ext = load_inline(
    name='sum_dim1_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    assert dim == 1
    # Result shape calculation: (Batch, D2)
    output = torch.zeros((x.shape[0], x.shape[2]), device=x.device, dtype=x.dtype)
    
    sum_ext.sum_dim1(x, output)
    
    # Reshape to expected (Batch, 1, D2)
    return output.unsqueeze(1)
