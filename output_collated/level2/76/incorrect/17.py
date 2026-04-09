# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_022315/code_0.py
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

# CUDA kernel implementation with memory coalescing optimization
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Optimized fused kernel with better memory coalescing
__global__ void fused_linear_bias_relu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M, int N, int K) 
{
    // Use 2D grid for better memory coalescing
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Batch dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Feature dimension
    
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Compute dot product with coalesced memory access
        // Both x and weight accesses are coalesced within warps
        for (int k = 0; k < K; ++k) {
            sum += x[row * K + k] * weight[col * K + k];
        }
        
        // Add bias and apply ReLU
        float result = sum + bias[col];
        output[row * N + col] = fmaxf(result, 0.0f);
    }
}

// Highly optimized tiled version with shared memory
__global__ void fused_linear_bias_relu_tiled_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M, int N, int K) 
{
    const int TILE_SIZE = 16;  // Reduced tile size for better fit in shared memory
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Calculate global indices
    const int row = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;
    
    // Shared memory tiles - ensure proper alignment
    __shared__ float x_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float w_tile[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;
    
    // Loop over K dimension in tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load x tile - coalesced access pattern
        int k = t * TILE_SIZE + tx;
        if (row < M && k < K) {
            x_tile[ty][tx] = x[row * K + k];
        } else {
            x_tile[ty][tx] = 0.0f;
        }
        
        // Load weight tile - coalesced access pattern  
        k = t * TILE_SIZE + ty;
        if (col < N && k < K) {
            w_tile[ty][tx] = weight[col * K + k];
        } else {
            w_tile[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product within tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += x_tile[ty][k] * w_tile[tx][k];  // Note: w_tile[tx][k] for transpose
        }
        
        __syncthreads();
    }
    
    // Write final result with bias and ReLU
    if (row < M && col < N) {
        float result = sum + bias[col];
        output[row * N + col] = fmaxf(result, 0.0f);
    }
}

void fused_linear_bias_relu(
    torch::Tensor x, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    torch::Tensor output) 
{
    const int M = x.size(0);  // batch size
    const int K = x.size(1);  // input features
    const int N = weight.size(0);  // output features
    
    // Check for valid tensor properties
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    
    // Configure kernel launch parameters
    const int TILE_SIZE = 16;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // Launch the tiled kernel for better performance
    fused_linear_bias_relu_tiled_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_linear_bias_relu(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_bias_relu", &fused_linear_bias_relu, "Fused Linear Bias ReLU with Memory Coalescing");
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
    # Output shape is (batch_size, out_features)
    batch_size = x.size(0)
    out_features = bias.size(0)
    output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_linear_bias_relu(x, gemm_weight, bias, output)
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
