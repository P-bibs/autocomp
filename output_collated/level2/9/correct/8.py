# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_071635/code_6.py
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

# CUDA Kernel for fused Linear + Sub + Mul + ReLU
# We use tiled GEMM to maintain efficiency over large matrix dimensions
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void fused_linear_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const float sub_val,
    const float mul_val,
    int batch, int in_f, int out_f) {

    __shared__ float x_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float w_tile[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int k = 0; k < (in_f + TILE_SIZE - 1) / TILE_SIZE; ++k) {
        if (row < batch && (k * TILE_SIZE + threadIdx.x) < in_f)
            x_tile[threadIdx.y][threadIdx.x] = x[row * in_f + k * TILE_SIZE + threadIdx.x];
        else
            x_tile[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < out_f && (k * TILE_SIZE + threadIdx.y) < in_f)
            w_tile[threadIdx.y][threadIdx.x] = weight[(k * TILE_SIZE + threadIdx.y) * out_f + col];
        else
            w_tile[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
            sum += x_tile[threadIdx.y][i] * w_tile[i][threadIdx.x];

        __syncthreads();
    }

    if (row < batch && col < out_f) {
        float val = sum + bias[col];
        val = (val - sub_val) * mul_val;
        out[row * out_f + col] = fmaxf(0.0f, val);
    }
}

void fused_op_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, 
                   torch::Tensor out, float sub, float mul) {
    int batch = x.size(0);
    int in_f = x.size(1);
    int out_f = weight.size(1);
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_f + TILE_SIZE - 1) / TILE_SIZE, (batch + TILE_SIZE - 1) / TILE_SIZE);
    fused_linear_kernel<<<grid, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), sub, mul, batch, in_f, out_f);
}
"""

cpp_source = r"""
void fused_op_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, 
                   torch::Tensor out, float sub, float mul);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_cuda, "Fused Linear-Subtract-Multiply-ReLU");
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
    out = torch.empty((x.size(0), linear_weight.size(0)), device=x.device, dtype=x.dtype)
    # Transpose weight because GEMM uses [In, Out] but Linear uses [Out, In]
    w_t = linear_weight.t() 
    fused_ext.fused_op(x, w_t, linear_bias, out, subtract_value, multiply_value)
    return out
