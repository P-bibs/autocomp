# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_104842/code_5.py
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

# The provided CUDA kernel uses a naive O(N^3) matrix multiplication approach.
# For high-performance on an RTX 2080Ti, we use shared memory tiling to ensure
# we stay memory bandwidth bound rather than compute bound by redundant DRAM access.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float mish(float x) {
    // Mish = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    // Using __expf, __logf, and __tanhf for fast math performance on RTX 2080Ti
    return x * tanhf(logf(1.0f + expf(x)));
}

template <int BLOCK_SIZE>
__global__ void fused_linear_mish_kernel(const float* __restrict__ input, 
                                        const float* __restrict__ weight, 
                                        const float* __restrict__ bias, 
                                        float* __restrict__ output,
                                        int batch_size, int in_features, int out_features) {
    __shared__ float sh_input[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sh_weight[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (in_features + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Load tiles into shared memory
        if (row < batch_size && (t * BLOCK_SIZE + threadIdx.x) < in_features)
            sh_input[threadIdx.y][threadIdx.x] = input[row * in_features + (t * BLOCK_SIZE + threadIdx.x)];
        else
            sh_input[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < out_features && (t * BLOCK_SIZE + threadIdx.y) < in_features)
            sh_weight[threadIdx.y][threadIdx.x] = weight[col * in_features + (t * BLOCK_SIZE + threadIdx.y)];
        else
            sh_weight[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += sh_input[threadIdx.y][k] * sh_weight[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < batch_size && col < out_features) {
        sum += bias[col];
        // Original code: Mish(Mish(x))
        output[row * out_features + col] = mish(mish(sum));
    }
}

void fused_linear_mish(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    const int B = input.size(0);
    const int I = input.size(1);
    const int O = weight.size(0);
    const int BLOCK_SIZE = 32;

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((O + BLOCK_SIZE - 1) / BLOCK_SIZE, (B + BLOCK_SIZE - 1) / BLOCK_SIZE);

    fused_linear_mish_kernel<BLOCK_SIZE><<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), B, I, O
    );
}
"""

cpp_source = r"""
void fused_linear_mish(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_mish", &fused_linear_mish, "Fused Linear + Double Mish kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, linear_weight, linear_bias):
    batch_size = x.size(0)
    out_features = linear_weight.size(0)
    output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
    fused_ext.fused_linear_mish(x, linear_weight, linear_bias, output)
    return output

# Inputs as required
batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda')]
