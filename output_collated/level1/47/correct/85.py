# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_141159/code_25.py
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
#  CUDA source – Vectorized kernel with explicit loop unrolling
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef UNROLL_FACTOR
#define UNROLL_FACTOR 8
#endif

__global__ void sum_dim1_float4_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       const int B, const int D1, const int D2) {
    const int tid = threadIdx.x;
    const int b = blockIdx.x;
    const int elements_per_thread = 4;
    const int elements_per_block = blockDim.x * elements_per_thread;
    const int j_base = blockIdx.y * elements_per_block + tid * elements_per_thread;

    if (b >= B || j_base >= D2) return;

    const bool aligned = (D2 % 4 == 0) && (j_base + 3 < D2);

    if (aligned) {
        float4 sum_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        int i = 0;
        
        #pragma unroll
        for ( ; i + UNROLL_FACTOR - 1 < D1; i += UNROLL_FACTOR) {
            #pragma unroll
            for (int u = 0; u < UNROLL_FACTOR; ++u) {
                const float4 val = __ldg((const float4*)&input[b * D1 * D2 + (i + u) * D2 + j_base]);
                sum_vec.x += val.x;
                sum_vec.y += val.y;
                sum_vec.z += val.z;
                sum_vec.w += val.w;
            }
        }
        for ( ; i < D1; ++i) {
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
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        int i = 0;
        
        #pragma unroll
        for ( ; i + UNROLL_FACTOR - 1 < D1; i += UNROLL_FACTOR) {
            #pragma unroll
            for (int u = 0; u < UNROLL_FACTOR; ++u) {
                const int stride = (i + u) * D2;
                if (j_base + 0 < D2) sum0 += __ldg(&input[b * D1 * D2 + stride + j_base + 0]);
                if (j_base + 1 < D2) sum1 += __ldg(&input[b * D1 * D2 + stride + j_base + 1]);
                if (j_base + 2 < D2) sum2 += __ldg(&input[b * D1 * D2 + stride + j_base + 2]);
                if (j_base + 3 < D2) sum3 += __ldg(&input[b * D1 * D2 + stride + j_base + 3]);
            }
        }
        for ( ; i < D1; ++i) {
            const int stride = i * D2;
            if (j_base + 0 < D2) sum0 += __ldg(&input[b * D1 * D2 + stride + j_base + 0]);
            if (j_base + 1 < D2) sum1 += __ldg(&input[b * D1 * D2 + stride + j_base + 1]);
            if (j_base + 2 < D2) sum2 += __ldg(&input[b * D1 * D2 + stride + j_base + 2]);
            if (j_base + 3 < D2) sum3 += __ldg(&input[b * D1 * D2 + stride + j_base + 3]);
        }
        
        if (j_base + 0 < D2) output[b * D2 + j_base + 0] = sum0;
        if (j_base + 1 < D2) output[b * D2 + j_base + 1] = sum1;
        if (j_base + 2 < D2) output[b * D2 + j_base + 2] = sum2;
        if (j_base + 3 < D2) output[b * D2 + j_base + 3] = sum3;
    }
}

void sum_dim1_float4(torch::Tensor input, torch::Tensor output) {
    const int B = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);
    const int threads = 256;
    const int elements_per_thread = 4;
    const int elements_per_block = threads * elements_per_thread;
    
    dim3 grid(B, (D2 + elements_per_block - 1) / elements_per_block);
    dim3 block(threads);

    sum_dim1_float4_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void sum_dim1_float4(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1_float4", &sum_dim1_float4, "Optimized sum along dimension 1");
}
"""

sum_ext = load_inline(
    name='sum_dim1_float4',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    assert dim == 1, "Only dimension 1 supported."
    B, D1, D2 = x.shape
    output = torch.zeros((B, D2), device=x.device, dtype=x.dtype)
    sum_ext.sum_dim1_float4(x, output)
    return output.unsqueeze(1)
