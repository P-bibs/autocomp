# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_021357/code_4.py
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

# CUDA kernel implementation with vectorization and improved GEMM
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_SIZE 32
#define WARP_SIZE 32

// Tiled GEMM kernel using float4 vectorization
// Computes: output = x @ weight.T
// x: (M, K), weight: (N, K), output: (M, N)
__global__ void gemm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int M, int K, int N) {
    
    // Shared memory for tiling
    __shared__ float tile_x[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_w[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Iterate over tiles of K dimension
    for (int tile_idx = 0; tile_idx < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile_idx) {
        int k_start = tile_idx * TILE_SIZE;
        int k_end = min(k_start + TILE_SIZE, K);
        int k_size = k_end - k_start;
        
        // Load tile from x: (M, K) -> tile_x[TILE_SIZE][TILE_SIZE]
        if (row < M && tx < k_size) {
            tile_x[ty][tx] = x[row * K + k_start + tx];
        } else {
            tile_x[ty][tx] = 0.0f;
        }
        
        // Load tile from weight: (N, K) -> tile_w[TILE_SIZE][TILE_SIZE]
        if (col < N && ty < k_size) {
            tile_w[ty][tx] = weight[col * K + k_start + ty];
        } else {
            tile_w[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int k = 0; k < k_size; ++k) {
            sum += tile_x[ty][k] * tile_w[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        output[row * N + col] = sum;
    }
}

// Vectorized bias + ReLU kernel using float4
__global__ void bias_relu_kernel_vectorized(
    float* __restrict__ output,
    const float* __restrict__ bias,
    int M, int N) {
    
    int total_elements = M * N;
    
    // Grid-stride loop with float4 vectorization
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < (total_elements / 4);
         idx += gridDim.x * blockDim.x) {
        
        // Compute which row and column this index corresponds to
        int element_idx = idx * 4;
        int row = element_idx / N;
        int col = element_idx % N;
        
        // Load 4 floats (must be aligned)
        float4 out_vals = *reinterpret_cast<float4*>(&output[element_idx]);
        float4 bias_vals;
        
        // Load bias values (broadcast if needed or load sequentially)
        bias_vals.x = bias[(col + 0) % N];
        bias_vals.y = bias[(col + 1) % N];
        bias_vals.z = bias[(col + 2) % N];
        bias_vals.w = bias[(col + 3) % N];
        
        // Add bias and apply ReLU
        out_vals.x = fmaxf(out_vals.x + bias_vals.x, 0.0f);
        out_vals.y = fmaxf(out_vals.y + bias_vals.y, 0.0f);
        out_vals.z = fmaxf(out_vals.z + bias_vals.z, 0.0f);
        out_vals.w = fmaxf(out_vals.w + bias_vals.w, 0.0f);
        
        // Store result
        *reinterpret_cast<float4*>(&output[element_idx]) = out_vals;
    }
    
    // Handle remaining elements (if total_elements is not divisible by 4)
    int remainder_start = (total_elements / 4) * 4;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int remainder_idx = remainder_start + tid;
    
    if (remainder_idx < total_elements) {
        int col = remainder_idx % N;
        output[remainder_idx] = fmaxf(output[remainder_idx] + bias[col], 0.0f);
    }
}

void fused_linear_bias_relu(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output)
{
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = weight.size(0);

    // 1. Perform GEMM with tiled kernel
    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim((N + TILE_SIZE - 1) / TILE_SIZE, 
                  (M + TILE_SIZE - 1) / TILE_SIZE);
    
    gemm_kernel<<<grid_dim, block_dim>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        M, K, N);

    // 2. Fused Bias + ReLU with vectorization
    int num_elements = M * N;
    int threads = 256;
    int blocks = min(65535, (num_elements + threads - 1) / threads);
    
    bias_relu_kernel_vectorized<<<blocks, threads>>>(
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        M, N);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_linear_bias_relu(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_bias_relu", &fused_linear_bias_relu, "Fused Linear Bias ReLU with vectorization");
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
        x: Input tensor (batch_size, in_features)
        gemm_weight: Weight tensor (out_features, in_features)
        bias: Bias tensor (out_features,)
    
    Returns:
        Output tensor (batch_size, out_features) with ReLU applied
    """
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
