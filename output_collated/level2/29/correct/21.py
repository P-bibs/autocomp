# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_105411/code_7.py
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
# CUDA source – fused linear + double mish kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BM 32
#define BN 32
#define BK 32

__device__ __forceinline__ float mish(float x) {
    // Mish: x * tanh(softplus(x)) = x * tanh(log1p(exp(x)))
    return x * tanhf(log1pf(expf(x)));
}

__global__ void fused_linear_mish_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K) 
{
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    int row = blockIdx.y * BM + threadIdx.y;
    int col = blockIdx.x * BN + threadIdx.x;
    float sum = 0.0f;

    for (int k = 0; k < (K + BK - 1) / BK; ++k) {
        if (row < M && (k * BK + threadIdx.x) < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + (k * BK + threadIdx.x)];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (k * BK + threadIdx.y) < K)
            sB[threadIdx.y][threadIdx.x] = B[(k * BK + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        float val = sum + bias[col];
        val = mish(val);
        val = mish(val);
        C[row * N + col] = val;
    }
}

void fused_op_cuda(const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& bias, torch::Tensor& out) {
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = weight.size(0);

    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    fused_linear_mish_kernel<<<grid, block>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), M, N, K
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_cuda(const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& bias, torch::Tensor& out);

torch::Tensor fused_op(const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& bias) {
    auto out = torch::empty({x.size(0), weight.size(0)}, x.options());
    fused_op_cuda(x, weight, bias, out);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused Linear + 2x Mish");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, linear_weight, linear_bias):
    # Ensure inputs are contiguous for CUDA access
    x = x.contiguous()
    w = linear_weight.t().contiguous() # Transposed for kernel access: N x K
    b = linear_bias.contiguous()
    return fused_ext.fused_op(x, w, b)

def get_init_inputs():
    return [8192, 8192]

def get_inputs():
    return [torch.rand(1024, 8192, device='cuda')]
