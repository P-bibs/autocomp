# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_020620/code_9.py
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

# The CUDA kernel uses Shared Memory Tiling to optimize global memory access.
# This replaces the naive implementation with a compute-bound matmul kernel.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void fused_linear_bias_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M, int N, int K
) {
    __shared__ float s_input[TILE_SIZE][TILE_SIZE];
    __shared__ float s_weight[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && (t * TILE_SIZE + tx) < K)
            s_input[ty][tx] = input[row * K + (t * TILE_SIZE + tx)];
        else
            s_input[ty][tx] = 0.0f;

        if (col < N && (t * TILE_SIZE + ty) < K)
            s_weight[ty][tx] = weight[(t * TILE_SIZE + ty) * N + col];
        else
            s_weight[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += s_input[ty][k] * s_weight[k][tx];

        __syncthreads();
    }

    if (row < M && col < N) {
        sum += bias[col];
        output[row * N + col] = fmaxf(0.0f, sum);
    }
}

void fused_linear_bias_relu_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output
) {
    const int M = input.size(0);
    const int K = input.size(1);
    const int N = weight.size(0);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    fused_linear_bias_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_linear_bias_relu_forward(const torch::Tensor input, const torch::Tensor weight, const torch::Tensor bias, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_bias_relu", &fused_linear_bias_relu_forward, "Fused Linear Bias ReLU");
}
"""

fused_ext = load_inline(
    name='fused_linear_bias_relu',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3'],
    with_cuda=True
)

def functional_model(x, *, gemm_weight, bias):
    # Transpose weight because weights in linear are typically [out, in] 
    # and matmul logic assumes weight matrix is [in, out] in standard form,
    # or handle the index logic accordingly. 
    # Standard linear: Y = XW^T + b. 
    w_t = gemm_weight.t() 
    output = torch.empty((x.size(0), w_t.size(1)), device=x.device, dtype=x.dtype)
    fused_ext.fused_linear_bias_relu(x, w_t, bias, output)
    return output

batch_size, in_features, out_features = 1024, 8192, 8192

def get_init_inputs():
    return [in_features, out_features, (out_features,)]

def get_inputs():
    # Return inputs on CUDA to ensure compatibility with the fused kernel
    return [torch.rand(batch_size, in_features, device='cuda')]
