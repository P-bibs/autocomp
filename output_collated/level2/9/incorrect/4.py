# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_073421/code_7.py
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

# -------------------------------------------------------------------------
# CUDA source – a tiled GEMM kernel fused with element-wise operations:
# (x @ W.T + b) * multiply_value - subtract_value, followed by ReLU.
# Note: F.linear computes x @ W.T + b.
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void fused_op_kernel(
    const float* __restrict__ A,   // M x K
    const float* __restrict__ W,   // N x K (weights transposed in linear input)
    const float* __restrict__ bias,// N
    float*       __restrict__ C,   // M x N
    int M, int N, int K,
    float mult,
    float sub
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;

    for (int kk = 0; kk < K; kk += TILE_SIZE) {
        // Load tile A[row, kk:kk+TILE_SIZE]
        if (row < M && (kk + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + (kk + threadIdx.x)];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile W[col, kk:kk+TILE_SIZE]
        if (col < N && (kk + threadIdx.y) < K) {
            Bs[threadIdx.y][threadIdx.x] = W[col * K + (kk + threadIdx.y)];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int l = 0; l < TILE_SIZE; ++l) {
            acc += As[threadIdx.y][l] * Bs[threadIdx.x][l];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        float val = (acc + bias[col]) * mult - sub;
        C[row * N + col] = (val > 0.0f) ? val : 0.0f;
    }
}

void fused_op_forward(
    const torch::Tensor& A,
    const torch::Tensor& W,
    const torch::Tensor& bias,
    torch::Tensor& C,
    float multiply_value,
    float subtract_value
) {
    int M = A.size(0);
    int K = A.size(1);
    int N = W.size(0);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    fused_op_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        W.data_ptr<float>(),
        bias.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        multiply_value,
        subtract_value
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const torch::Tensor& A,
    const torch::Tensor& W,
    const torch::Tensor& bias,
    torch::Tensor& C,
    float multiply_value,
    float subtract_value
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused linear + scale + sub + ReLU forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    linear_weight,
    linear_bias,
    subtract_value,
    multiply_value,
):
    # Ensure inputs are contiguous float32 on GPU
    x = x.to(device='cuda', dtype=torch.float32).contiguous()
    weight = linear_weight.to(device='cuda', dtype=torch.float32).contiguous()
    bias = linear_bias.to(device='cuda', dtype=torch.float32).contiguous()
    
    out = torch.empty((x.size(0), weight.size(0)), device='cuda', dtype=torch.float32)
    
    fused_ext.fused_op(x, weight, bias, out, multiply_value, subtract_value)
    return out
