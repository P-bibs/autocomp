# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_020620/code_0.py
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

# CUDA kernel implementation with optimized memory coalescing
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// Fused GEMM + Bias + ReLU kernel with tiling for better memory coalescing
__global__ void fused_gemm_bias_relu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M, int N, int K) {
    
    // Shared memory for tiles
    __shared__ float tile_x[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_w[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    // Tile-based computation for better memory coalescing
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory with bounds checking
        int x_col = t * TILE_SIZE + threadIdx.x;
        int w_row = t * TILE_SIZE + threadIdx.y;
        
        // Load x tile
        if (row < M && x_col < K) {
            tile_x[threadIdx.y][threadIdx.x] = x[row * K + x_col];
        } else {
            tile_x[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load weight tile (weight is transposed in computation)
        if (w_row < K && col < N) {
            tile_w[threadIdx.y][threadIdx.x] = weight[col * K + w_row];
        } else {
            tile_w[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_x[threadIdx.y][k] * tile_w[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result with bias addition and ReLU
    if (row < M && col < N) {
        float val = sum + bias[col];
        output[row * N + col] = fmaxf(0.0f, val);
    }
}

// Alternative implementation with grid-stride loop for better register usage
__global__ void fused_gemm_bias_relu_kernel_v2(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M, int N, int K) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop to handle large matrices efficiently
    for (int r = row; r < M; r += blockDim.y * gridDim.y) {
        for (int c = col; c < N; c += blockDim.x * gridDim.x) {
            float sum = 0.0f;
            
            // Compute dot product with coalesced memory access
            for (int k = 0; k < K; ++k) {
                sum += x[r * K + k] * weight[c * K + k];
            }
            
            // Apply bias and ReLU
            float val = sum + bias[c];
            output[r * N + c] = fmaxf(0.0f, val);
        }
    }
}

void fused_op_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output) 
{
    const int M = x.size(0);  // batch_size
    const int K = x.size(1);  // in_features
    const int N = weight.size(0);  // out_features
    
    // Use tiled version for better memory coalescing when dimensions are large
    if (M >= 512 && N >= 512) {
        // Configure kernel launch parameters for tiled version
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        
        fused_gemm_bias_relu_kernel<<<grid, block>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    } else {
        // Configure kernel launch parameters for grid-stride version
        dim3 block(16, 16);
        dim3 grid(min(65535, (N + 15) / 16), min(65535, (M + 15) / 16));
        
        fused_gemm_bias_relu_kernel_v2<<<grid, block>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K
        );
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused GEMM + Bias + ReLU");
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
    batch_size = x.size(0)
    out_features = bias.size(0)
    output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(x, gemm_weight, bias, output)
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
