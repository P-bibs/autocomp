# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110448/code_4.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_SIZE 32

__device__ __forceinline__ float mish(float x) {
    // Mish: x * tanh(softplus(x))
    // softplus(x) = ln(1 + exp(x))
    // Use stable softplus
    float s = (x > 20.0f) ? x : logf(1.0f + expf(x));
    return x * tanhf(s);
}

__global__ void fused_linear_double_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features) {

    __shared__ float s_input[TILE_SIZE][TILE_SIZE];
    __shared__ float s_weight[TILE_SIZE][TILE_SIZE];

    int batch_idx = blockIdx.x * TILE_SIZE + threadIdx.y;
    int out_idx = blockIdx.y * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;

    for (int k_tile = 0; k_tile < in_features; k_tile += TILE_SIZE) {
        // Load tiles into shared memory
        if (batch_idx < batch_size && (k_tile + threadIdx.x) < in_features)
            s_input[threadIdx.y][threadIdx.x] = input[batch_idx * in_features + k_tile + threadIdx.x];
        else
            s_input[threadIdx.y][threadIdx.x] = 0.0f;

        if (out_idx < out_features && (k_tile + threadIdx.y) < in_features)
            s_weight[threadIdx.y][threadIdx.x] = weight[out_idx * in_features + k_tile + threadIdx.y];
        else
            s_weight[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += s_input[threadIdx.y][k] * s_weight[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (batch_idx < batch_size && out_idx < out_features) {
        float val = acc + (bias ? bias[out_idx] : 0.0f);
        val = mish(mish(val));
        output[batch_idx * out_features + out_idx] = val;
    }
}

void fused_linear_double_mish_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_features,
    int out_features) {

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((batch_size + TILE_SIZE - 1) / TILE_SIZE, (out_features + TILE_SIZE - 1) / TILE_SIZE);
    
    fused_linear_double_mish_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_features, out_features
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_linear_double_mish_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int batch_size, int in_features, int out_features);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_double_mish", &fused_linear_double_mish_forward, "Fused Linear + 2x Mish");
}
"""

fused_ext = load_inline(
    name='fused_linear_double_mish',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, linear_weight, linear_bias):
    batch_size = x.shape[0]
    in_features = x.shape[1]
    out_features = linear_weight.shape[0]
    output = torch.empty(batch_size, out_features, device=x.device, dtype=x.dtype)
    fused_ext.fused_linear_double_mish(x, linear_weight, linear_bias, output, batch_size, in_features, out_features)
    return output
