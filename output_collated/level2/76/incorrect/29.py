# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_024220/code_4.py
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

# CUDA kernel implementation with fused GEMM + Bias + ReLU
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_SIZE 32

// Fused GEMM + Bias + ReLU kernel
// Computes: output = ReLU(x @ weight.T + bias)
// x: (M, K), weight: (N, K), bias: (N), output: (M, N)
__global__ void fused_gemm_bias_relu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M, int N, int K)
{
    // Shared memory for tiling
    __shared__ float tile_x[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_weight[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Each thread accumulates one output element
    float acc = 0.0f;
    
    // Iterate through tiles of the K dimension
    for (int tile_k = 0; tile_k < K; tile_k += TILE_SIZE) {
        // Load tile_x: (TILE_SIZE x TILE_SIZE) from x
        // x is stored as (M, K) in row-major order
        if (row < M && (tile_k + threadIdx.x) < K) {
            tile_x[threadIdx.y][threadIdx.x] = x[row * K + (tile_k + threadIdx.x)];
        } else {
            tile_x[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile_weight: (TILE_SIZE x TILE_SIZE) from weight
        // weight is stored as (N, K) in row-major order
        if (col < N && (tile_k + threadIdx.y) < K) {
            tile_weight[threadIdx.y][threadIdx.x] = weight[col * K + (tile_k + threadIdx.y)];
        } else {
            tile_weight[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += tile_x[threadIdx.y][k] * tile_weight[threadIdx.x][k];
        }
        
        __syncthreads();
    }
    
    // Write result with bias and ReLU applied
    if (row < M && col < N) {
        float val = acc + bias[col];
        output[row * N + col] = fmaxf(val, 0.0f);
    }
}

void fused_gemm_bias_relu(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output)
{
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = weight.size(0);
    
    // Grid and block configuration
    // Each block handles TILE_SIZE x TILE_SIZE output elements
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);
    dim3 threads(TILE_SIZE, TILE_SIZE);
    
    fused_gemm_bias_relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_gemm_bias_relu(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gemm_bias_relu", &fused_gemm_bias_relu,
          "Fused GEMM + Bias + ReLU operation");
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
    Fused linear layer with bias and ReLU activation.
    
    Args:
        x: Input tensor of shape (batch_size, in_features)
        gemm_weight: Weight tensor of shape (out_features, in_features)
        bias: Bias tensor of shape (out_features,)
    
    Returns:
        Output tensor of shape (batch_size, out_features) with ReLU applied
    """
    batch_size = x.size(0)
    out_features = bias.size(0)
    
    # Allocate output tensor
    output = torch.empty(
        (batch_size, out_features),
        device=x.device,
        dtype=x.dtype
    )
    
    # Call fused GEMM + Bias + ReLU kernel
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
    gemm_weight = torch.rand(out_features, in_features, device='cuda')
    bias = torch.rand(out_features, device='cuda')
    return [x], {"gemm_weight": gemm_weight, "bias": bias}
