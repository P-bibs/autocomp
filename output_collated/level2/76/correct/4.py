# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_014309/code_7.py
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

# -------------------------------------------------------------------------
# High-Performance Fused Kernel
# - Tiled matrix multiplication utilizing Shared Memory (tiling 32x32x32)
# - Fused Bias Addition and ReLU activation
# - Coalesced memory access patterns
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

template <typename scalar_t>
__global__ void fused_linear_relu_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int M, const int N, const int K) 
{
    __shared__ scalar_t x_tile[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t w_tile[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    scalar_t sum = 0;

    // Tiled Matrix Multiplication
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load x_tile
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K)
            x_tile[threadIdx.y][threadIdx.x] = x[row * K + (t * TILE_SIZE + threadIdx.x)];
        else
            x_tile[threadIdx.y][threadIdx.x] = 0;

        // Load w_tile (weight is [N, K], we want weight^T for col-major-like access)
        if (col < N && (t * TILE_SIZE + threadIdx.y) < K)
            w_tile[threadIdx.y][threadIdx.x] = weight[col * K + (t * TILE_SIZE + threadIdx.y)];
        else
            w_tile[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += x_tile[threadIdx.y][k] * w_tile[k][threadIdx.x];
        }
        __syncthreads();
    }

    // Fused bias and ReLU
    if (row < M && col < N) {
        scalar_t val = sum + bias[col];
        output[row * N + col] = (val > 0) ? val : 0;
    }
}

void fused_op_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output)
{
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = weight.size(0);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_linear_relu", ([&] {
        fused_linear_relu_kernel<scalar_t><<<grid, block>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            M, N, K);
    }));
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Linear+Bias+ReLU kernel");
}
"""

# Compile extension
fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, gemm_weight, bias):
    # Ensure inputs are contiguous and on GPU
    x = x.contiguous().cuda()
    weight = gemm_weight.contiguous().cuda()
    bias = bias.contiguous().cuda()
    
    output = torch.empty((x.size(0), weight.size(0)), dtype=x.dtype, device=x.device)
    fused_ext.fused_op(x, weight, bias, output)
    return output

# Constants for evaluation harness
batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)

def get_init_inputs():
    return [in_features, out_features, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_features)]
