# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_065701/code_5.py
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

# CUDA kernel with shared memory tiling to improve bandwidth efficiency
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void fused_linear_kernel(
    const float* __restrict__ x, const float* __restrict__ weight, 
    const float* __restrict__ bias, float* __restrict__ out,
    int M, int N, int K, float sub, float mul) {
    
    __shared__ float s_x[TILE_SIZE][TILE_SIZE];
    __shared__ float s_w[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K)
            s_x[threadIdx.y][threadIdx.x] = x[row * K + t * TILE_SIZE + threadIdx.x];
        else
            s_x[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (t * TILE_SIZE + threadIdx.y) < K)
            s_w[threadIdx.y][threadIdx.x] = weight[col * K + t * TILE_SIZE + threadIdx.y];
        else
            s_w[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            acc += s_x[threadIdx.y][k] * s_w[k][threadIdx.x];
        
        __syncthreads();
    }

    if (row < M && col < N) {
        acc += bias[col];
        acc = (acc - sub) * mul;
        out[row * N + col] = fmaxf(acc, 0.0f);
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, 
                      torch::Tensor out, float sub, float mul) {
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = weight.size(0);
    
    const dim3 threads(TILE_SIZE, TILE_SIZE);
    const dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    fused_linear_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.data_ptr<float>(), out.data_ptr<float>(), 
        M, N, K, sub, mul);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, 
                      torch::Tensor out, float sub, float mul);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused linear, sub, mul, relu kernel");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, linear_weight, linear_bias, subtract_value, multiply_value):
    # Expects linear_weight as (out_features, in_features)
    out = torch.empty((x.shape[0], linear_weight.shape[0]), device=x.device, dtype=torch.float32)
    fused_ext.fused_op(x, linear_weight, linear_bias, out, subtract_value, multiply_value)
    return out
