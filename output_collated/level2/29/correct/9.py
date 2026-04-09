# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_104237/code_5.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float mish(float x) {
    // Stability fix: Use softplus approximation or standard library
    // Mish(x) = x * tanhf(logf(1.0f + expf(x)))
    // For x > 20, tanhf(approx) is 1.0. For x < -20, result is near 0.
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return x * tanhf(logf(1.0f + expf(x)));
}

// Tiled matrix multiplication fused with activation
template <int BLOCK_SIZE>
__global__ void fused_linear_mish_kernel(const float* __restrict__ input, 
                                        const float* __restrict__ weight, 
                                        const float* __restrict__ bias, 
                                        float* __restrict__ output, 
                                        int batch_size, int in_features, int out_features) {
    __shared__ float tile_input[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_weight[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int k_step = 0; k_step < (in_features + BLOCK_SIZE - 1) / BLOCK_SIZE; ++k_step) {
        int k = k_step * BLOCK_SIZE;
        
        if (row < batch_size && (k + threadIdx.x) < in_features)
            tile_input[threadIdx.y][threadIdx.x] = input[row * in_features + k + threadIdx.x];
        else
            tile_input[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < out_features && (k + threadIdx.y) < in_features)
            tile_weight[threadIdx.y][threadIdx.x] = weight[col * in_features + k + threadIdx.y];
        else
            tile_weight[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += tile_input[threadIdx.y][i] * tile_weight[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < batch_size && col < out_features) {
        sum += bias[col];
        float m = mish(sum);
        output[row * out_features + col] = mish(m);
    }
}

void fused_linear_mish(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    const int BLOCK_SIZE = 16;
    int batch_size = input.size(0);
    int out_features = weight.size(0);
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((out_features + BLOCK_SIZE - 1) / BLOCK_SIZE, (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    fused_linear_mish_kernel<BLOCK_SIZE><<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
        output.data_ptr<float>(), batch_size, input.size(1), out_features
    );
}
"""

cpp_source = r"""
void fused_linear_mish(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_mish", &fused_linear_mish, "Fused Linear + Mish + Mish");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, linear_weight, linear_bias):
    out = torch.empty((x.shape[0], linear_weight.shape[0]), device=x.device, dtype=x.dtype)
    fused_ext.fused_linear_mish(x, linear_weight, linear_bias, out)
    return out

batch_size, in_features, out_features = 1024, 8192, 8192
def get_init_inputs(): return [in_features, out_features]
def get_inputs(): return [torch.rand(batch_size, in_features, device='cuda')]
