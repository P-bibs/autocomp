# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_111334/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['linear_weight', 'linear_bias']
REQUIRED_FLAT_STATE_NAMES = ['linear_weight', 'linear_bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Mish, and applies Mish again.
    """

    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

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

# Optimization: Tile operations to fuse GEMM and Mish activations.
# By performing the dot product within a tile and applying activation 
# before writing to global memory, we minimize global memory bandwidth.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 32

__device__ __forceinline__ float mish(float x) {
    // Standard Mish: x * tanh(ln(1 + exp(x)))
    // We use log1p for numerical stability: log(1 + exp(x)) = log1p(exp(x))
    // But since CUDA has expf and logf, we'll stick to those with max for stability
    float exp_x = expf(x);
    return x * tanhf(logf(1.0f + exp_x));
}

__global__ void fused_linear_mish_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int M,
    int N,
    int K
) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;
    
    __shared__ float tile_x[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_w[TILE_SIZE][TILE_SIZE];

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile of x
        int x_row = row;
        int x_col = t * TILE_SIZE + threadIdx.x;
        if (x_row < M && x_col < K) {
            tile_x[threadIdx.y][threadIdx.x] = x[x_row * K + x_col];
        } else {
            tile_x[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile of w (transposed in shared memory for coalesced access)
        int w_row = t * TILE_SIZE + threadIdx.y;
        int w_col = col;
        if (w_row < K && w_col < N) {
            tile_w[threadIdx.y][threadIdx.x] = w[w_row * N + w_col];
        } else {
            tile_w[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        for (int i = 0; i < TILE_SIZE; ++i) {
            acc += tile_x[threadIdx.y][i] * tile_w[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Apply bias and activation functions
    if (row < M && col < N) {
        float val = acc + b[col];
        // Apply Mish twice
        val = mish(val);
        val = mish(val);
        out[row * N + col] = val;
    }
}

void fused_op_forward_kernel(
    const torch::Tensor x,
    const torch::Tensor w,
    const torch::Tensor b,
    torch::Tensor out
) {
    int M = x.size(0);  // batch size
    int K = x.size(1);  // input features
    int N = w.size(0);  // output features
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    fused_linear_mish_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N, K
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the function in the .cu file
void fused_op_forward_kernel(
    const torch::Tensor x,
    const torch::Tensor w,
    const torch::Tensor b,
    torch::Tensor out
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward_kernel, "Fused Linear + Mish operator");
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

def functional_model(x, *, linear_weight, linear_bias):
    out = torch.empty((x.shape[0], linear_weight.shape[0]), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, linear_weight, linear_bias, out)
    return out

batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features)]
