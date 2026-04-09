# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110448/code_5.py
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
    if (x > 20.0f) return x; 
    if (x < -20.0f) return expf(x);
    return x * tanhf(logf(1.0f + expf(x)));
}

#define TILE_SIZE 32

__global__ void fused_linear_mish_kernel(const float* __restrict__ x, const float* __restrict__ w, 
                                        const float* __restrict__ b, float* __restrict__ out, 
                                        int B, int N, int M) {
    __shared__ float s_x[TILE_SIZE][TILE_SIZE];
    __shared__ float s_w[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float acc = 0.0f;
    for (int k = 0; k < (N + TILE_SIZE - 1) / TILE_SIZE; ++k) {
        if (row < B && (k * TILE_SIZE + threadIdx.x) < N)
            s_x[threadIdx.y][threadIdx.x] = x[row * N + (k * TILE_SIZE + threadIdx.x)];
        else
            s_x[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < M && (k * TILE_SIZE + threadIdx.y) < N)
            s_w[threadIdx.y][threadIdx.x] = w[col * N + (k * TILE_SIZE + threadIdx.y)];
        else
            s_w[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
            acc += s_x[threadIdx.y][i] * s_w[threadIdx.x][i];

        __syncthreads();
    }

    if (row < B && col < M) {
        float res = acc + b[col];
        float m1 = mish(res);
        out[row * M + col] = mish(m1);
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out) {
    int B = x.size(0);
    int N = x.size(1);
    int M = w.size(0);
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((M + TILE_SIZE - 1) / TILE_SIZE, (B + TILE_SIZE - 1) / TILE_SIZE);
    fused_linear_mish_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), B, N, M);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Linear and Mish activation");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, linear_weight, linear_bias):
    out = torch.empty((x.size(0), linear_weight.size(0)), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, linear_weight, linear_bias, out)
    return out

batch_size, in_features, out_features = 1024, 8192, 8192
def get_init_inputs(): return [in_features, out_features]
def get_inputs(): return [torch.rand(batch_size, in_features).cuda()]
