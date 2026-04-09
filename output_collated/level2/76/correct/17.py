# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_021357/code_11.py
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

# Tiled CUDA kernel for GEMM + Bias + ReLU
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void fused_gemm_bias_relu_kernel(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b, float* __restrict__ out, int M, int N, int K) {
    __shared__ float s_x[TILE_SIZE][TILE_SIZE];
    __shared__ float s_w[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int k_tile = 0; k_tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++k_tile) {
        if (row < M && (k_tile * TILE_SIZE + threadIdx.x) < K)
            s_x[threadIdx.y][threadIdx.x] = x[row * K + (k_tile * TILE_SIZE + threadIdx.x)];
        else
            s_x[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (k_tile * TILE_SIZE + threadIdx.y) < K)
            s_w[threadIdx.y][threadIdx.x] = w[col * K + (k_tile * TILE_SIZE + threadIdx.y)];
        else
            s_w[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_x[threadIdx.y][k] * s_w[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        float val = sum + b[col];
        out[row * N + col] = (val > 0.0f) ? val : 0.0f;
    }
}

void fused_op(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out) {
    int M = x.size(0);
    int N = w.size(0);
    int K = x.size(1);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    fused_gemm_bias_relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), 
        out.data_ptr<float>(), M, N, K
    );
}
"""

cpp_source = r"""
void fused_op(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused GEMM Bias ReLU Kernel");
}
"""

# Compile the extension
fused_module = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, gemm_weight, bias):
    # Prepare output tensor
    out = torch.empty((x.shape[0], gemm_weight.shape[0]), device=x.device, dtype=x.dtype)
    # Launch fused kernel
    fused_module.fused_op(x, gemm_weight, bias, out)
    return out

# Global vars for compatibility
batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)

def get_init_inputs():
    return [in_features, out_features, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]
