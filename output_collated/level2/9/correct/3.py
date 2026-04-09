# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_065701/code_4.py
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

# Tiled CUDA kernel for high-performance matrix multiplication + fusion
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void fused_linear_act_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float subtract_value,
    const float multiply_value,
    const int N, // batch_size
    const int M, // in_features
    const int K  // out_features
) {
    __shared__ float s_input[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_weight[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (M + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (row < N && (t * TILE_WIDTH + threadIdx.x) < M)
            s_input[threadIdx.y][threadIdx.x] = input[row * M + t * TILE_WIDTH + threadIdx.x];
        else
            s_input[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < K && (t * TILE_WIDTH + threadIdx.y) < M)
            s_weight[threadIdx.y][threadIdx.x] = weight[col * M + t * TILE_WIDTH + threadIdx.y];
        else
            s_weight[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += s_input[threadIdx.y][k] * s_weight[k][threadIdx.x];
        
        __syncthreads();
    }

    if (row < N && col < K) {
        sum += bias[col];
        sum = fmaxf(0.0f, (sum - subtract_value) * multiply_value);
        output[row * K + col] = sum;
    }
}

void fused_linear_act_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output, float sub, float mul
) {
    int N = input.size(0); // batch
    int M = input.size(1); // in
    int K = weight.size(0); // out

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((K + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    fused_linear_act_kernel<<<dimGrid, dimBlock>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), sub, mul, N, M, K
    );
}
"""

cpp_source = r"""
void fused_linear_act_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, float sub, float mul);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_act_forward", &fused_linear_act_forward, "Fused linear act");
}
"""

fused_ext = load_inline(
    name='fused_linear_act', cpp_sources=cpp_source, cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, linear_weight, linear_bias, subtract_value, multiply_value):
    batch_size = x.size(0)
    out_features = linear_weight.size(0)
    output = torch.empty(batch_size, out_features, device='cuda', dtype=torch.float32)
    fused_ext.fused_linear_act_forward(x.cuda(), linear_weight.cuda(), linear_bias.cuda(), output, subtract_value, multiply_value)
    return output
