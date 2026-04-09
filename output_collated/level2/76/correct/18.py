# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_021357/code_12.py
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

# CUDA kernel implementation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// Block-Tiled GEMM: Compute Y = X @ W^T
// Utilizing shared memory for x and weight (transposed) tiles
__global__ void gemm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int M, int K, int N) {
    
    __shared__ float sh_x[TILE_SIZE][TILE_SIZE];
    __shared__ float sh_w[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float acc = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load x tile: [M, K]
        int k_idx = t * TILE_SIZE + tx;
        sh_x[ty][tx] = (row < M && k_idx < K) ? x[row * K + k_idx] : 0.0f;
        
        // Load weight tile (weight is N x K, effectively W^T is K x N)
        // We transpose logic here to use weight as [N, K]
        int w_col = blockIdx.x * TILE_SIZE + tx;
        int w_row = t * TILE_SIZE + ty;
        sh_w[ty][tx] = (w_col < N && w_row < K) ? weight[w_col * K + w_row] : 0.0f;
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += sh_x[ty][k] * sh_w[k][tx];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        output[row * N + col] = acc;
    }
}

// Vectorized Bias + ReLU: Uses float4 for 4x memory throughput
__global__ void bias_relu_vectorized(
    float* __restrict__ data,
    const float* __restrict__ bias,
    int M, int N) {
    
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int total_elements = M * N;
    
    if (idx < total_elements) {
        float4* data_vec = reinterpret_cast<float4*>(&data[idx]);
        float4 val = *data_vec;
        
        // Broadcast bias for 4 indices
        // Note: Assumes N is large/multiple of 4. Handling modulo if necessary.
        int col = idx % N;
        if (col + 3 < N) {
            float4 b = *reinterpret_cast<const float4*>(&bias[col]);
            val.x = fmaxf(val.x + b.x, 0.0f);
            val.y = fmaxf(val.y + b.y, 0.0f);
            val.z = fmaxf(val.z + b.z, 0.0f);
            val.w = fmaxf(val.w + b.w, 0.0f);
            *data_vec = val;
        } else {
            // Scalar fallback for boundary
            for(int i=0; i<4 && (idx+i) < total_elements; ++i) {
                data[idx+i] = fmaxf(data[idx+i] + bias[(idx+i)%N], 0.0f);
            }
        }
    }
}

void fused_linear_bias_relu(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = weight.size(0);
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    gemm_kernel<<<grid, block>>>(x.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), M, K, N);
    
    int num_vec_threads = (M * N + 3) / 4;
    int threads = 256;
    int blocks = (num_vec_threads + threads - 1) / threads;
    bias_relu_vectorized<<<blocks, threads>>>(output.data_ptr<float>(), bias.data_ptr<float>(), M, N);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_linear_bias_relu(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_bias_relu", &fused_linear_bias_relu, "Fused GEMM + Bias + ReLU");
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
