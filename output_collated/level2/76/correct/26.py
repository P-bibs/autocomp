# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_023135/code_14.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'bias_shape']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['gemm_weight', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['gemm_weight', 'bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, adds a bias term, and applies ReLU.
    """

    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

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
    # State for gemm (nn.Linear)
    if 'gemm_weight' in flat_state:
        state_kwargs['gemm_weight'] = flat_state['gemm_weight']
    else:
        state_kwargs['gemm_weight'] = getattr(model.gemm, 'weight', None)
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
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

# -----------------------------------------------------------------------------
# CUDA Kernel: Tiled GEMM with Fused Bias and ReLU
# Tiling strategy: BM=16, BN=16, BK=16
# -----------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BM 16
#define BN 16
#define BK 16

__global__ void gemm_bias_relu_kernel(
    const float* __restrict__ A,    // x: (M, K)
    const float* __restrict__ B,    // weight: (N, K)
    const float* __restrict__ bias, // bias: (N)
    float* __restrict__ C,          // output: (M, N)
    int M, int N, int K) 
{
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    int row = blockIdx.y * BM + threadIdx.y;
    int col = blockIdx.x * BN + threadIdx.x;

    float acc = 0.0f;

    // Tiling over K dimension
    for (int k = 0; k < K; k += BK) {
        // Load A tile
        if (row < M && (k + threadIdx.x) < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + (k + threadIdx.x)];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile (Transposed access to B to match weight(N, K))
        // We want to access weight[col][k + threadIdx.y]
        if (col < N && (k + threadIdx.y) < K)
            sB[threadIdx.y][threadIdx.x] = B[col * K + (k + threadIdx.y)];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product
        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            acc += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }

    // Fused bias and ReLU write-back
    if (row < M && col < N) {
        float val = acc + bias[col];
        C[row * N + col] = (val > 0.0f) ? val : 0.0f;
    }
}

void fused_gemm_bias_relu(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = weight.size(0);

    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    gemm_bias_relu_kernel<<<grid, block>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.data_ptr<float>(), output.data_ptr<float>(), 
        M, N, K);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_gemm_bias_relu(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gemm_bias_relu", &fused_gemm_bias_relu, "Fused GEMM Bias ReLU Kernel");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, gemm_weight, bias):
    """
    Computes ReLU(x @ gemm_weight.T + bias) using a custom fused CUDA kernel.
    x: (M, K), gemm_weight: (N, K), bias: (N)
    """
    batch_size = x.shape[0]
    out_features = bias.shape[0]
    output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_gemm_bias_relu(x, gemm_weight, bias, output)
    return output
