# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_071635/code_5.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'subtract_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['linear_weight', 'linear_bias', 'subtract_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['linear_weight', 'linear_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a matrix multiplication, subtraction, multiplication, and ReLU activation.
    """

    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

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
    if 'subtract_value' in flat_state:
        state_kwargs['subtract_value'] = flat_state['subtract_value']
    else:
        state_kwargs['subtract_value'] = getattr(model, 'subtract_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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

# The custom CUDA kernel implements a Tiled GEMM followed by element-wise bias addition,
# subtraction, scaling, and ReLU activation. This avoids writing the intermediate GEMM
# result to Global Memory, significantly reducing memory bandwidth pressure.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 32

__global__ void fused_op_kernel(const float* __restrict__ x, const float* __restrict__ weight, 
                                const float* __restrict__ bias, float* __restrict__ out,
                                float sub_val, float mul_val, int M, int N, int K) {
    // Shared memory for tiling input x and weight matrix
    __shared__ float s_x[TILE_DIM][TILE_DIM];
    __shared__ float s_w[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float sum = 0.0f;

    // Tiled matrix multiplication
    for (int k_tile = 0; k_tile < K; k_tile += TILE_DIM) {
        if (row < M && (k_tile + threadIdx.x) < K)
            s_x[threadIdx.y][threadIdx.x] = x[row * K + k_tile + threadIdx.x];
        else
            s_x[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (k_tile + threadIdx.y) < K)
            s_w[threadIdx.y][threadIdx.x] = weight[(k_tile + threadIdx.y) * N + col];
        else
            s_w[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            sum += s_x[threadIdx.y][k] * s_w[k][threadIdx.x];
        }
        __syncthreads();
    }

    // Element-wise fused operations
    if (row < M && col < N) {
        float res = (sum + bias[col] - sub_val) * mul_val;
        out[row * N + col] = res > 0.0f ? res : 0.0f;
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, 
                      torch::Tensor out, float sub, float mul) {
    int M = x.size(0); 
    int K = x.size(1); 
    int N = weight.size(1);
    
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    
    fused_op_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
        out.data_ptr<float>(), sub, mul, M, N, K);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, 
                      torch::Tensor out, float sub, float mul);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused GEMM + Elementwise (Bias, Sub, Mul, ReLU)");
}
"""

# Compile the CUDA extension
fused_ext = load_inline(
    name='fused_op', 
    cpp_sources=cpp_source, 
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], 
    with_cuda=True
)

def functional_model(x, *, linear_weight, linear_bias, subtract_value, multiply_value):
    # PyTorch linear layer typically expects (N, K), we pass it as (K, N) to the kernel
    # linear_weight is (N, K), transform to (K, N)
    weight_t = linear_weight.t().contiguous()
    out = torch.empty((x.size(0), linear_weight.size(0)), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(x, weight_t, linear_bias, out, subtract_value, multiply_value)
    return out
