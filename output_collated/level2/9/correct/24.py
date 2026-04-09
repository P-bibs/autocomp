# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_075823/code_6.py
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

# The custom CUDA kernel implements a Tiled Matrix Multiplication
# combined with the element-wise operations (subtract, multiply, relu).
# A block size of 32x32 is used for tiling to balance memory access 
# and shared memory utilization.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 32

__global__ void fused_linear_ops_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    float sub_val,
    float mul_val
) {
    __shared__ float s_x[TILE_DIM][TILE_DIM];
    __shared__ float s_w[TILE_DIM][TILE_DIM];

    int batch_idx = blockIdx.y * TILE_DIM + threadIdx.y;
    int out_idx = blockIdx.x * TILE_DIM + threadIdx.x;

    float acc = 0.0f;
    for (int k = 0; k < (in_features + TILE_DIM - 1) / TILE_DIM; ++k) {
        // Load x tile
        if (batch_idx < batch_size && (k * TILE_DIM + threadIdx.x) < in_features)
            s_x[threadIdx.y][threadIdx.x] = x[batch_idx * in_features + (k * TILE_DIM + threadIdx.x)];
        else
            s_x[threadIdx.y][threadIdx.x] = 0.0f;

        // Load weight tile (weight is [out_features, in_features])
        if (out_idx < out_features && (k * TILE_DIM + threadIdx.y) < in_features)
            s_w[threadIdx.y][threadIdx.x] = weight[out_idx * in_features + (k * TILE_DIM + threadIdx.y)];
        else
            s_w[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_DIM; ++i) {
            acc += s_x[threadIdx.y][i] * s_w[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (batch_idx < batch_size && out_idx < out_features) {
        float val = (acc + bias[out_idx] - sub_val) * mul_val;
        output[batch_idx * out_features + out_idx] = (val > 0.0f) ? val : 0.0f;
    }
}

void fused_linear_ops_launch(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output, float sub_val, float mul_val
) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((out_features + TILE_DIM - 1) / TILE_DIM, (batch_size + TILE_DIM - 1) / TILE_DIM);

    fused_linear_ops_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_features, out_features, sub_val, mul_val
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_linear_ops_launch(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, float sub_val, float mul_val);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_ops", &fused_linear_ops_launch, "Fused Linear + Bias + Sub + Mul + ReLU");
}
"""

fused_ext = load_inline(
    name='fused_linear_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3'],
    with_cuda=True
)

def functional_model(x, *, linear_weight, linear_bias, subtract_value, multiply_value):
    output = torch.empty(x.shape[0], linear_weight.shape[0], device=x.device, dtype=x.dtype)
    fused_ext.fused_linear_ops(x, linear_weight, linear_bias, output, float(subtract_value), float(multiply_value))
    return output
