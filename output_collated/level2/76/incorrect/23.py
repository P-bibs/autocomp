# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_023135/code_10.py
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

# Optimized CUDA kernel with register tiling for compute efficiency
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Using 16x16 thread block, each thread computes 4x4 output tile
#define TILE_M 64
#define TILE_N 64
#define TILE_K 16

__global__ void fused_tiled_gemm_bias_relu_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    const float* __restrict__ bias, 
    float* __restrict__ C, 
    int M, int N, int K) 
{
    __shared__ float sA[TILE_M][TILE_K];
    __shared__ float sB[TILE_K][TILE_N];

    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    float acc[4][4] = {0.0f};

    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; ++t) {
        // Load A and B into shared memory
        for(int i = 0; i < TILE_M; i += blockDim.y)
            for(int j = 0; j < TILE_K; j += blockDim.x)
                sA[threadIdx.y + i][threadIdx.x + j] = A[min(row + i, M - 1) * K + (t * TILE_K + threadIdx.x + j)];

        for(int i = 0; i < TILE_K; i += blockDim.y)
            for(int j = 0; j < TILE_N; j += blockDim.x)
                sB[threadIdx.y + i][threadIdx.x + j] = B[(t * TILE_K + threadIdx.y + i) * N + min(col + j, N - 1)];

        __syncthreads();

        for (int k = 0; k < TILE_K; ++k) {
            #pragma unroll
            for(int i = 0; i < 4; ++i)
                for(int j = 0; j < 4; ++j)
                    acc[i][j] += sA[threadIdx.y + i * 16][k] * sB[k][threadIdx.x + j * 16];
        }
        __syncthreads();
    }

    #pragma unroll
    for(int i = 0; i < 4; ++i) {
        for(int j = 0; j < 4; ++j) {
            int r = row + i * 16;
            int c = col + j * 16;
            if (r < M && c < N) {
                float val = acc[i][j] + bias[c];
                C[r * N + c] = (val > 0.0f) ? val : 0.0f;
            }
        }
    }
}

void fused_linear_bias_relu(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    int M = x.size(0);
    int K = x.size(1);
    int N = weight.size(0);
    dim3 threads(16, 16);
    dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    fused_tiled_gemm_bias_relu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), M, N, K);
}
"""

cpp_source = r"""
void fused_linear_bias_relu(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_bias_relu", &fused_linear_bias_relu, "Fused Linear Bias ReLU");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, gemm_weight, bias):
    batch_size = x.size(0)
    out_features = bias.size(0)
    output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
    fused_ext.fused_linear_bias_relu(x, gemm_weight, bias, output)
    return output
