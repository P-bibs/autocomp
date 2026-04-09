# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_023135/code_9.py
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

# CUDA kernel with Tiled GEMM + Bias + ReLU
# Tiling effectively reduces global memory traffic by reusing blocks of data in shared memory
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void fused_gemm_bias_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_features,
    const int out_features) {
    
    __shared__ float s_input[TILE_SIZE][TILE_SIZE];
    __shared__ float s_weight[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;

    for (int k = 0; k < (in_features + TILE_SIZE - 1) / TILE_SIZE; ++k) {
        // Load tiles into shared memory
        if (row < batch_size && (k * TILE_SIZE + threadIdx.x) < in_features)
            s_input[threadIdx.y][threadIdx.x] = input[row * in_features + k * TILE_SIZE + threadIdx.x];
        else
            s_input[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < out_features && (k * TILE_SIZE + threadIdx.y) < in_features)
            s_weight[threadIdx.y][threadIdx.x] = weight[col * in_features + (k * TILE_SIZE + threadIdx.y)];
        else
            s_weight[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += s_input[threadIdx.y][i] * s_weight[threadIdx.x][i];
        }
        __syncthreads();
    }

    if (row < batch_size && col < out_features) {
        float val = sum + bias[col];
        output[row * out_features + col] = (val > 0.0f) ? val : 0.0f;
    }
}

void fused_gemm_bias_relu_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output) {
    
    const int batch_size = input.size(0);
    const int out_features = weight.size(0);
    const int in_features = input.size(1);
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_features + TILE_SIZE - 1) / TILE_SIZE, (batch_size + TILE_SIZE - 1) / TILE_SIZE);
    
    fused_gemm_bias_relu_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
}
"""

cpp_source = r"""
void fused_gemm_bias_relu_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gemm_bias_relu", &fused_gemm_bias_relu_forward, "Fused Tiled GEMM + Bias + ReLU");
}
"""

fused_ext = load_inline(
    name='fused_gemm_bias_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, gemm_weight, bias):
    # Output: [batch_size, out_features]
    # Weight is usually [out_features, in_features] for linear
    output = torch.empty(x.size(0), gemm_weight.size(0), device=x.device, dtype=x.dtype)
    fused_ext.fused_gemm_bias_relu(x, gemm_weight, bias, output)
    return output
