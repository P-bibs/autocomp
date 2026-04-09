# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110005/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['linear_weight', 'linear_bias']
REQUIRED_FLAT_STATE_NAMES = ['linear_weight', 'linear_bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Mish, and applies Mish again.
    """

    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

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
    # State for linear (nn.Linear)
    if 'linear_weight' in flat_state:
        state_kwargs['linear_weight'] = flat_state['linear_weight']
    else:
        state_kwargs['linear_weight'] = getattr(model.linear, 'weight', None)
    if 'linear_bias' in flat_state:
        state_kwargs['linear_bias'] = flat_state['linear_bias']
    else:
        state_kwargs['linear_bias'] = getattr(model.linear, 'bias', None)
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

# ----------------------------------------------------------------------
# CUDA source – WMMA-based GEMM with fused double-mish activation
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// Mish activation: x * tanh(softplus(x))
__device__ __forceinline__ float mish_float(float x) {
    float sp = log1pf(expf(-fabsf(x))) + fmaxf(x, 0.0f); // Numerically stable softplus
    return x * tanhf(sp);
}

/* 
   Optimized CUDA Kernel:
   - Performs multiplication A (N, K) * B_T (M, K)^T + bias
   - Uses WMMA 16x16x16 Tensor Cores
   - Applies Mish twice in the register file
   - Writes directly to global memory C (N, M)
*/
__global__ void gemm_mish_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    const float* __restrict__ bias,
    half* __restrict__ C,
    int N, int K, int M) 
{
    int row_offset = blockIdx.y * 16;
    int col_offset = blockIdx.x * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += 16) {
        if (row_offset < N && k < K)
            wmma::load_matrix_sync(a_frag, A + row_offset * K + k, K);
        if (col_offset < M && k < K)
            wmma::load_matrix_sync(b_frag, B + col_offset * K + k, K);
        
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    int tid = threadIdx.x;
    int i = tid / 16;
    int j = tid % 16;

    if (row_offset + i < N && col_offset + j < M) {
        float val = c_frag.x[i * 16 + j];
        if (bias != nullptr)
            val += bias[col_offset + j];
        
        val = mish_float(val);
        val = mish_float(val);
        
        C[(row_offset + i) * M + (col_offset + j)] = __float2half(val);
    }
}

void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor bias, torch::Tensor C) {
    int N = A.size(0);
    int K = A.size(1);
    int M = B.size(0);
    
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    dim3 block(256);

    gemm_mish_kernel<<<grid, block>>>(
        (const half*)A.data_ptr(), (const half*)B.data_ptr(), 
        bias.defined() ? (const float*)bias.data_ptr() : nullptr, 
        (half*)C.data_ptr(), N, K, M
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor bias, torch::Tensor C);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused FP16 GEMM Mish kernel");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '-arch=sm_75', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, linear_weight, linear_bias):
    # Prepare inputs: Weights are M x K (transposed for efficient WMMA col_major load)
    w_t = linear_weight.t().contiguous().half()
    x_half = x.contiguous().half()
    
    batch_size, out_features = x.size(0), linear_weight.size(0)
    out = torch.empty((batch_size, out_features), dtype=torch.float16, device=x.device)
    
    bias = linear_bias.float() if linear_bias is not None else torch.tensor([], dtype=torch.float32, device=x.device)
    
    fused_ext.fused_op(x_half, w_t, bias, out)
    
    return out.float()
