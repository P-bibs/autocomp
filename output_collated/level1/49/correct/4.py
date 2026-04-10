# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150540/code_1.py
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
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void max_kernel(const float* input, float* output, int N, int M, int K) {
    // N: batch_size, M: dim1, K: dim2. Reducing along dim=1 (M)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * K;

    for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
        int n = i / K;
        int k = i % K;
        float max_val = -1e38f; // Sufficient for float range
        const float* input_row = &input[n * M * K + k];
        for (int m = 0; m < M; ++m) {
            float val = input_row[m * K];
            if (val > max_val) max_val = val;
        }
        output[i] = max_val;
    }
}

void max_forward_kernelLauncher(const float* input, float* output, int N, int M, int K) {
    int threads = 256;
    int blocks = (N * K + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    
    max_kernel<<<blocks, threads>>>(input, output, N, M, K);
}
"""

cpp_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

void max_forward_kernelLauncher(const float* input, float* output, int N, int M, int K);

torch::Tensor max_forward(torch::Tensor x) {
    int N = x.size(0);
    int M = x.size(1);
    int K = x.size(2);
    auto output = torch::empty({N, K}, x.options());
    
    max_forward_kernelLauncher(x.data_ptr<float>(), output.data_ptr<float>(), N, M, K);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_forward", &max_forward, "Max reduction forward pass");
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
    # Only supports dim=1 reduction based on problem scope
    if dim != 1:
        return torch.max(x, dim=dim)[0]
    return fused_ext.max_forward(x.contiguous())

# Example usage for verification
if __name__ == "__main__":
    batch_size = 128
    dim1 = 4096
    dim2 = 4095
    x = torch.rand(batch_size, dim1, dim2).cuda()
    res = functional_model(x, dim=1)
    print("Success, output shape:", res.shape)
