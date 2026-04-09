# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_104237/code_4.py
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

# Tiled GEMM kernel with fused Mish activations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float mish(float x) {
    // Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    // Using fast math approximation via expf and tanhf
    return x * tanhf(log1pf(expf(x)));
}

#define TILE_SIZE 16

__global__ void fused_linear_double_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M, int K, int N
) {
    __shared__ float s_input[TILE_SIZE][TILE_SIZE];
    __shared__ float s_weight[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K)
            s_input[threadIdx.y][threadIdx.x] = input[row * K + (t * TILE_SIZE + threadIdx.x)];
        else
            s_input[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (t * TILE_SIZE + threadIdx.y) < K)
            s_weight[threadIdx.y][threadIdx.x] = weight[col * K + (t * TILE_SIZE + threadIdx.y)];
        else
            s_weight[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
            sum += s_input[threadIdx.y][i] * s_weight[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N) {
        sum += bias[col];
        sum = mish(mish(sum));
        output[row * N + col] = sum;
    }
}

void launch_fused(const torch::Tensor input, const torch::Tensor weight, const torch::Tensor bias, torch::Tensor output) {
    int M = input.size(0);
    int K = input.size(1);
    int N = weight.size(0);
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    fused_linear_double_mish_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
        output.data_ptr<float>(), M, K, N);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_fused(const torch::Tensor input, const torch::Tensor weight, const torch::Tensor bias, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_fused, "Fused Linear + Double Mish");
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
    batch_size, out_features = x.size(0), linear_weight.size(0)
    output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, linear_weight, linear_bias, output)
    return output

batch_size, in_features, out_features = 1024, 8192, 8192
def get_init_inputs(): return [in_features, out_features]
def get_inputs(): return [torch.rand(batch_size, in_features).cuda()]
