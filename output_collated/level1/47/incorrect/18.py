# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_143434/code_16.py
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
#include <vector_types.h>

// Vectorized reduction kernel: Each thread processes 4 elements along the reducing dimension (D1)
// and handles one output column using float4 loads for D2 efficiency.
__global__ void sum_dim1_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                int B, int D1, int D2) {
    int b = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (b >= B || j >= (D2 / 4)) return;

    // Use float4 to load 4 elements along the D2 dimension
    // Each thread writes to 4 columns of the output
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    
    const int col_idx = j * 4;
    const float* batch_ptr = input + (b * D1 * D2) + col_idx;

    for (int i = 0; i < D1; ++i) {
        const float4* data_ptr = reinterpret_cast<const float4*>(batch_ptr + (i * D2));
        float4 val = *data_ptr;
        sum0 += val.x;
        sum1 += val.y;
        sum2 += val.z;
        sum3 += val.w;
    }

    float4 res = make_float4(sum0, sum1, sum2, sum3);
    float4* out_ptr = reinterpret_cast<float4*>(output + (b * D2) + col_idx);
    *out_ptr = res;
}

void sum_dim1(torch::Tensor input, torch::Tensor output) {
    const int B = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);
    
    // 256 threads is stable for high occupancy.
    // D2/4 is the number of float4 blocks to process
    dim3 threads(256);
    dim3 blocks(B, (D2 / 4 + threads.x - 1) / threads.x);
    
    sum_dim1_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), B, D1, D2
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void sum_dim1(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1", &sum_dim1, "Optimized sum along dim 1 using float4 load/store");
}
"""

sum_ext = load_inline(
    name='sum_dim1',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(x, *, dim):
    assert dim == 1
    # D2 must be divisible by 4 for the float4 vectorized kernel
    assert x.shape[2] % 4 == 0, "D2 must be multiple of 4"
    
    if not x.is_contiguous():
        x = x.contiguous()
        
    output = torch.zeros((x.shape[0], x.shape[2]), device=x.device, dtype=x.dtype)
    sum_ext.sum_dim1(x, output)
    return output.unsqueeze(1)
