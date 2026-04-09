# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_024220/code_3.py
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

# CUDA kernel source
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void fused_op_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int batch_size,
    int in_features,
    int out_features
) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;

    // Loop over tiles
    for (int k = 0; k < (in_features + TILE_SIZE - 1) / TILE_SIZE; ++k) {
        // Load tile of A (x)
        if (row < batch_size && (k * TILE_SIZE + threadIdx.x) < in_features) {
            sA[threadIdx.y][threadIdx.x] = x[row * in_features + k * TILE_SIZE + threadIdx.x];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile of B (weight)
        if (col < out_features && (k * TILE_SIZE + threadIdx.y) < in_features) {
            sB[threadIdx.y][threadIdx.x] = weight[col * in_features + k * TILE_SIZE + threadIdx.y];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial multiplication
        for (int i = 0; i < TILE_SIZE; ++i) {
            acc += sA[threadIdx.y][i] * sB[threadIdx.x][i]; // Note: sB is transposed in layout
        }

        __syncthreads();
    }

    // Apply bias and ReLU
    if (row < batch_size && col < out_features) {
        float val = acc + bias[col];
        out[row * out_features + col] = fmaxf(0.0f, val);
    }
}

void fused_op_forward(
    const torch::Tensor x,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor out,
    int batch_size,
    int in_features,
    int out_features
) {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((out_features + TILE_SIZE - 1) / TILE_SIZE, (batch_size + TILE_SIZE - 1) / TILE_SIZE);
    
    fused_op_forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
}
"""

# C++ binding source
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const torch::Tensor x,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor out,
    int batch_size,
    int in_features,
    int out_features
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused GEMM + Bias + ReLU forward pass");
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
    batch_size = x.shape[0]
    in_features = x.shape[1]
    out_features = gemm_weight.shape[0]
    
    # Transpose weight to match kernel expectation (out_features x in_features)
    weight_t = gemm_weight.contiguous()
    
    # Allocate output tensor
    out = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
    
    # Launch fused kernel
    fused_ext.fused_op(x, weight_t, bias, out, batch_size, in_features, out_features)
    
    return out

# Input shape definitions
batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)

def get_init_inputs():
    return [in_features, out_features, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_features)]
