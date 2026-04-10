# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_143434/code_23.py
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

# --- CUDA Kernel ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Each thread reduces a column D2 across D1
__global__ void sum_reduction_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                     int B, int D1, int D2) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.y;
    
    if (col < D2) {
        float sum = 0.0f;
        // Access pattern: input[batch][0][col], input[batch][1][col]...
        // This is non-coalesced if D2 is not 1, but we can improve 
        // by reading a block of floats if D2 was small.
        // Given D2=4095, we perform sequential loading per column.
        const float* batch_ptr = input + (batch * D1 * D2) + col;
        for (int i = 0; i < D1; ++i) {
            sum += batch_ptr[i * D2];
        }
        output[(batch * D2) + col] = sum;
    }
}

void launch_sum(torch::Tensor input, torch::Tensor output, int B, int D1, int D2) {
    dim3 threads(32); // Use warp size for better occupancy
    dim3 blocks((D2 + threads.x - 1) / threads.x, B);
    sum_reduction_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), B, D1, D2);
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>
void launch_sum(torch::Tensor input, torch::Tensor output, int B, int D1, int D2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_sum", &launch_sum, "Coalesced reduction of dim 1");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='sum_reduction',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    assert dim == 1, "This optimization specifically targets dim=1 reduction"
    B, D1, D2 = x.shape
    # Output shape for keepdim=True is (B, 1, D2)
    output = torch.empty((B, 1, D2), device=x.device, dtype=x.dtype)
    
    # Flatten the middle dimension for the kernel view
    # The output expects (B, 1, D2), we pass as (B, D2) to map col index
    fused_ext.launch_sum(x.contiguous(), output.view(B, D2), B, D1, D2)
    return output

# Verification/Usage
if __name__ == "__main__":
    batch_size, dim1, dim2 = 128, 4096, 4095
    x = torch.rand(batch_size, dim1, dim2).cuda()
    
    # Execution
    result = functional_model(x, dim=1)
    
    # Validation against PyTorch native
    expected = torch.sum(x, dim=1, keepdim=True)
    assert torch.allclose(result, expected, atol=1e-5)
    print("Verification successful!")
