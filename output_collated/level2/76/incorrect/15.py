# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_021357/code_10.py
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

# CUDA kernel with Shared Memory Tiling for optimized performance
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 32

__global__ void tiled_linear_bias_relu_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    int M, int K, int N) 
{
    __shared__ float s_x[BLOCK_DIM][BLOCK_DIM];
    __shared__ float s_w[BLOCK_DIM][BLOCK_DIM];

    int row = blockIdx.y * BLOCK_DIM + threadIdx.y;
    int col = blockIdx.x * BLOCK_DIM + threadIdx.x;

    float acc = 0.0f;

    // Loop over tiles of K
    for (int t = 0; t < (K + BLOCK_DIM - 1) / BLOCK_DIM; ++t) {
        // Load tile into shared memory
        if (row < M && (t * BLOCK_DIM + threadIdx.x) < K)
            s_x[threadIdx.y][threadIdx.x] = x[row * K + (t * BLOCK_DIM + threadIdx.x)];
        else
            s_x[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (t * BLOCK_DIM + threadIdx.y) < K)
            s_w[threadIdx.y][threadIdx.x] = weight[col * K + (t * BLOCK_DIM + threadIdx.y)];
        else
            s_w[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Perform dot product on tile
        for (int k = 0; k < BLOCK_DIM; ++k) {
            acc += s_x[threadIdx.y][k] * s_w[threadIdx.x][k];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        float val = acc + bias[col];
        output[row * N + col] = (val > 0.0f) ? val : 0.0f;
    }
}

void fused_linear_bias_relu(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = weight.size(0);

    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks((N + BLOCK_DIM - 1) / BLOCK_DIM, (M + BLOCK_DIM - 1) / BLOCK_DIM);

    tiled_linear_bias_relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        M, K, N
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_linear_bias_relu(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_bias_relu", &fused_linear_bias_relu, "Tiled Fused Linear Bias ReLU");
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
    """
    Computes (x @ gemm_weight.T) + bias with ReLU activation.
    Optimized via shared-memory tiled matrix multiplication.
    """
    batch_size = x.size(0)
    out_features = bias.size(0)
    output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_linear_bias_relu(x, gemm_weight, bias, output)
    return output
