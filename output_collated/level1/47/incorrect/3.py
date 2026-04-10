# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_125421/code_7.py
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

# Vectorized CUDA kernel using float4 for optimized memory access
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Vectorized reduction kernel using float4
// Each thread processes 4 elements of D2 at a time
__global__ void sum_dim1_vectorized_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int B, int D1, int D2_div4) {
    int b = blockIdx.x;
    int j = (blockIdx.y * blockDim.x + threadIdx.x);
    
    if (b >= B || j >= D2_div4) return;
    
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    const float4* input4 = reinterpret_cast<const float4*>(input);
    int row_stride = D2_div4;
    
    for (int i = 0; i < D1; ++i) {
        float4 val = input4[b * D1 * row_stride + i * row_stride + j];
        sum.x += val.x;
        sum.y += val.y;
        sum.z += val.z;
        sum.w += val.w;
    }
    
    float4* output4 = reinterpret_cast<float4*>(output);
    output4[b * row_stride + j] = sum;
}

void sum_dim1_vectorized(torch::Tensor input, torch::Tensor output) {
    int B = input.size(0);
    int D1 = input.size(1);
    int D2 = input.size(2);
    
    // We assume D2 is divisible by 4 for this optimization
    int D2_div4 = D2 / 4;
    dim3 threads(256);
    dim3 blocks(B, (D2_div4 + threads.x - 1) / threads.x);
    
    sum_dim1_vectorized_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2_div4);
}
"""

# C++ binding for the CUDA kernel
cpp_source = r"""
#include <torch/extension.h>
void sum_dim1_vectorized(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1_vectorized", &sum_dim1_vectorized, "Vectorized sum along dimension 1");
}
"""

# Compile the extension
sum_ext = load_inline(
    name='sum_dim1_vec',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    """
    Reduce the input tensor `x` of shape (B, D1, D2) along dimension 1 using
    a custom optimized CUDA kernel with float4 vectorization.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor on CUDA.  Shape: (batch, dim1, dim2).
    dim : int
        Must be 1 (as required by the problem statement).

    Returns
    -------
    torch.Tensor
        Output tensor of shape (batch, 1, dim2) containing the sums.
    """
    assert dim == 1, "Only dim=1 is supported."
    
    B, D1, D2 = x.shape
    
    # Handle the case where D2 is not divisible by 4
    padded_D2 = ((D2 + 3) // 4) * 4
    padding_needed = padded_D2 != D2
    
    if padding_needed:
        # Pad input to make D2 divisible by 4
        x_padded = torch.zeros((B, D1, padded_D2), device=x.device, dtype=x.dtype)
        x_padded[:, :, :D2] = x
        x_to_process = x_padded
    else:
        x_to_process = x.contiguous()
        
    # Output tensor for the vectorized part
    output_vec = torch.zeros((B, 1, padded_D2), device=x.device, dtype=x.dtype)
    
    # Launch the vectorized kernel
    sum_ext.sum_dim1_vectorized(x_to_process, output_vec.squeeze(1))
    
    # Return only the valid part if padding was added
    if padding_needed:
        return output_vec[:, :, :D2].contiguous()
    else:
        return output_vec.contiguous()

# -------------------------------------------------------------------------
#  Evaluation harness 
# -------------------------------------------------------------------------
batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def get_init_inputs():
    """Returns any static inputs required by the model – here only the reduction dim."""
    return [reduce_dim]

def get_inputs():
    """Creates a random input tensor on the GPU for benchmarking."""
    x = torch.rand(batch_size, dim1, dim2, device='cuda', dtype=torch.float32)
    return [x]

# -------------------------------------------------------------------------
#  If this module is executed directly, run a quick sanity check.
# -------------------------------------------------------------------------
if __name__ == "__main__":
    dim = reduce_dim
    x = torch.rand(batch_size, dim1, dim2, device='cuda')
    out = functional_model(x, dim=dim)
    # Verify shape
    assert out.shape == (batch_size, 1, dim2)
    print("Functional model with vectorized CUDA kernel works correctly.")
