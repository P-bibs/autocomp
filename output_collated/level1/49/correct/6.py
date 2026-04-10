# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150540/code_5.py
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

# Optimization: Grid-stride loop kernel for Max Reduction
# The kernel maps the 2D output (batch_size, dim2) to threads.
# Each thread performs a grid-stride loop reduction across the dim1 dimension.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void max_kernel(const float* __restrict__ input, float* __restrict__ output, int N, int M, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = N * K;

    for (int i = idx; i < total_output_elements; i += blockDim.x * gridDim.x) {
        int n = i / K;
        int k = i % K;
        float max_val = -FLT_MAX; 
        
        // Offset into the input tensor for the current (n, k) slice
        // Input layout: [N, M, K]. Elements are contiguous in K.
        const float* input_ptr = input + (n * M * K) + k;
        
        for (int m = 0; m < M; ++m) {
            float val = input_ptr[m * K];
            if (val > max_val) max_val = val;
        }
        output[i] = max_val;
    }
}

torch::Tensor max_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    
    int N = x.size(0);
    int M = x.size(1);
    int K = x.size(2);
    auto output = torch::empty({N, K}, x.options());
    
    int total_threads = N * K;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // Cap to max grid size for safety
    
    max_kernel<<<blocks, threads>>>(x.data_ptr<float>(), output.data_ptr<float>(), N, M, K);
    
    return output;
}
"""

cpp_source = r"""
#include <torch/extension.h>

torch::Tensor max_forward(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_forward", &max_forward, "Max reduction along dim 1");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='max_module',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    """
    Optimized functional model using custom CUDA kernel for max reduction.
    """
    if dim != 1:
        # Fallback for dimensions other than 1 as per constraints
        return torch.max(x, dim=dim)[0]
    
    # Ensure memory is contiguous for direct pointer access in CUDA
    if not x.is_contiguous():
        x = x.contiguous()
        
    return fused_ext.max_forward(x)

# Example usage/Sanity check
if __name__ == "__main__":
    batch_size = 128
    dim1 = 4096
    dim2 = 4095
    x = torch.rand(batch_size, dim1, dim2).cuda()
    res = functional_model(x, dim=1)
    expected = torch.max(x, dim=1)[0]
    
    # Verify numerical correctness
    diff = torch.abs(res - expected).max()
    print(f"Max difference: {diff.item()}")
    print("Optimization complete. Output shape:", res.shape)
