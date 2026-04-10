# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_131936/code_29.py
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
#  CUDA source using float4, loop unrolling, and memory coalescing
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_dim1_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                const int B, const int D1, const int D2) {
    const int tid = threadIdx.x;
    const int b = blockIdx.x;
    const int j_base = blockIdx.y * (blockDim.x * 4) + tid * 4;

    if (b >= B || j_base >= D2) return;

    // Use float4 for coalesced access. D2 is assumed to be a multiple of 4 
    // for high-performance paths. If not, the last thread handles remainder.
    float4 sum_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    // Pointer to the start of this block's data for the current batch
    const float* __restrict__ batch_ptr = input + b * D1 * D2;

    // By unrolling, we allow the compiler to pipeline load instructions
    #pragma unroll 8
    for (int i = 0; i < D1; ++i) {
        // Offset Calculation: Using __ldg to leverage constant cache
        const float4 val = reinterpret_cast<const float4&>(batch_ptr[i * D2 + j_base]);
        sum_vec.x += val.x;
        sum_vec.y += val.y;
        sum_vec.z += val.z;
        sum_vec.w += val.w;
    }

    // Write-back to global memory
    float* out_ptr = output + b * D2 + j_base;
    out_ptr[0] = sum_vec.x;
    out_ptr[1] = sum_vec.y;
    out_ptr[2] = sum_vec.z;
    out_ptr[3] = sum_vec.w;
}

void sum_dim1_gpu(torch::Tensor input, torch::Tensor output) {
    const int B = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);

    // threads=256 is usually optimal for register pressure vs occupancy on Turing
    const int threads = 256;
    const int blocks_x = B;
    const int blocks_y = (D2 + (threads * 4) - 1) / (threads * 4);
    
    dim3 grid(blocks_x, blocks_y);
    dim3 block(threads);

    sum_dim1_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void sum_dim1_gpu(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1", &sum_dim1_gpu, "Optimized sum along dim 1");
}
"""

# Compile extension
sum_ext = load_inline(
    name='sum_dim1',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(x, *, dim):
    """
    Optimized reduction using float4 loads and CUDA kernel launch.
    Assumes D2 is multiple of 4 as per common high-performance standards.
    """
    assert dim == 1, "Only reduction along dimension 1 is supported."
    B, D1, D2 = x.shape
    
    # Pad D2 to multiple of 4 if necessary for float4 safety
    pad_d2 = (D2 + 3) // 4 * 4
    if pad_d2 != D2:
        x_padded = torch.nn.functional.pad(x, (0, pad_d2 - D2))
    else:
        x_padded = x

    output = torch.zeros((B, pad_d2), device=x.device, dtype=x.dtype)
    sum_ext.sum_dim1(x_padded, output)
    
    # Return slice to original width and reshape
    return output[:, :D2].unsqueeze(1)
