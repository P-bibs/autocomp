# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_075823/code_1.py
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

# CUDA Kernel: Fused GEMM (Tiled) + Bias + Subtract + Multiply + ReLU
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void fused_op_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int N, int K, int M,
    float subtract_value,
    float multiply_value
) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;
    
    // Tiled matrix multiplication
    for (int k = 0; k < (K + TILE_SIZE - 1) / TILE_SIZE; ++k) {
        __shared__ float sh_x[TILE_SIZE][TILE_SIZE];
        __shared__ float sh_w[TILE_SIZE][TILE_SIZE];

        int x_k = k * TILE_SIZE + threadIdx.x;
        int w_k = k * TILE_SIZE + threadIdx.y;

        // Load tiles into shared memory
        sh_x[threadIdx.y][threadIdx.x] = (row < N && x_k < K) ? x[row * K + x_k] : 0.0f;
        sh_w[threadIdx.y][threadIdx.x] = (col < M && w_k < K) ? w[col * K + w_k] : 0.0f;

        __syncthreads();

        // Compute partial dot product
        for (int i = 0; i < TILE_SIZE; ++i) {
            acc += sh_x[threadIdx.y][i] * sh_w[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Apply bias, subtract, multiply and ReLU
    if (row < N && col < M) {
        float val = acc + b[col];
        val = (val - subtract_value) * multiply_value;
        out[row * M + col] = fmaxf(0.0f, val);
    }
}

void fused_op_forward(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor out,
    float subtract_value,
    float multiply_value
) {
    int N = x.size(0);  // batch size
    int K = x.size(1);  // in_features
    int M = w.size(0);  // out_features
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    fused_op_forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        N, K, M,
        subtract_value,
        multiply_value
    );
}
"""

# C++ Logic (Interface/Bindings)
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the function in the .cu file
void fused_op_forward(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor out,
    float subtract_value,
    float multiply_value
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused linear, subtract, multiply, and ReLU operation");
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

def functional_model(x, *, linear_weight, linear_bias, subtract_value, multiply_value):
    # Ensure inputs are on the correct device and dtype
    out = torch.empty((x.size(0), linear_weight.size(0)), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, linear_weight, linear_bias, out, subtract_value, multiply_value)
    return out

batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]

def get_inputs():
    return [torch.rand(batch_size, in_features)]
