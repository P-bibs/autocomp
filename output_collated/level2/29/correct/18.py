# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_105411/code_4.py
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

# We use tiled matrix multiplication to compute Linear + Double Mish in one kernel.
# This avoids the O(N^2) global memory bandwidth bottleneck of a naive kernel.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 32

__device__ __forceinline__ float mish(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return x * tanhf(logf(1.0f + expf(x)));
}

__global__ void fused_linear_mish_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features
) {
    __shared__ float s_input[TILE_DIM][TILE_DIM];
    __shared__ float s_weight[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float acc = (row < out_features) ? bias[row] : 0.0f;

    for (int t = 0; t < (in_features + TILE_DIM - 1) / TILE_DIM; ++t) {
        if (row < out_features && (t * TILE_DIM + threadIdx.x) < in_features)
            s_weight[threadIdx.y][threadIdx.x] = weight[row * in_features + (t * TILE_DIM + threadIdx.x)];
        else
            s_weight[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < batch_size && (t * TILE_DIM + threadIdx.y) < in_features)
            s_input[threadIdx.y][threadIdx.x] = input[(t * TILE_DIM + threadIdx.y) * batch_size + col];
        else
            s_input[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            acc += s_weight[threadIdx.y][k] * s_input[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < out_features && col < batch_size) {
        float m1 = mish(acc);
        float m2 = mish(m1);
        output[col * out_features + row] = m2;
    }
}

void fused_op_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_features,
    int out_features
) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((batch_size + TILE_DIM - 1) / TILE_DIM, (out_features + TILE_DIM - 1) / TILE_DIM);
    
    fused_linear_mish_mish_kernel<<<grid, block>>>(
        input.transpose(0, 1).contiguous().data_ptr<float>(),
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
void fused_op_forward(const torch::Tensor input, const torch::Tensor weight, const torch::Tensor bias, torch::Tensor output, int batch_size, int in_features, int out_features);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused linear + double mish");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, linear_weight, linear_bias):
    batch_size, in_features = x.shape
    out_features = linear_weight.shape[0]
    output = torch.empty(batch_size, out_features, device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, linear_weight, linear_bias, output, batch_size, in_features, out_features)
    return output
