# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_023135/code_12.py
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

# Optimized CUDA kernel implementation
# Using a 2D tiled GEMM approach with shared memory and fused ReLU/Bias
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void fused_gemm_bias_relu_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    const float* __restrict__ bias, 
    float* __restrict__ C, 
    int M, int K, int N) 
{
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;
    for (int k_tile = 0; k_tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++k_tile) {
        if (row < M && k_tile * TILE_SIZE + threadIdx.x < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + k_tile * TILE_SIZE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && k_tile * TILE_SIZE + threadIdx.y < K)
            sB[threadIdx.y][threadIdx.x] = B[(k_tile * TILE_SIZE + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        float val = acc + bias[col];
        C[row * N + col] = (val > 0.0f) ? val : 0.0f;
    }
}

void fused_gemm_bias_relu(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = weight.size(0);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    fused_gemm_bias_relu_kernel<<<grid, block>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        M, K, N
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_gemm_bias_relu(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gemm_bias_relu", &fused_gemm_bias_relu, "Fused GEMM Bias ReLU");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, gemm_weight, bias):
    # Transpose weight inside to match GEMM expectations if necessary, 
    # but since problem expects (N, K) layout for weight, we feed it directly.
    batch_size = x.size(0)
    out_features = bias.size(0)
    output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
    
    # Kernel assumes weight is (N, K) as per standard GEMM logic (O = X @ W.T)
    fused_ext.fused_gemm_bias_relu(x, gemm_weight, bias, output)
    return output

# Parameters for evaluation
batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features, (out_features,)]

def get_inputs():
    x = torch.rand(batch_size, in_features, device='cuda')
    # gemm_weight set to (out_features, in_features) to conform to the original logic expectations
    gemm_weight = torch.rand(out_features, in_features, device='cuda')
    bias = torch.rand(out_features, device='cuda')
    return [x], {"gemm_weight": gemm_weight, "bias": bias}
