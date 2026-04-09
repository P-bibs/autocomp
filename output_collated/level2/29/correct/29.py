# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110448/code_7.py
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

# ----------------------------------------------------------------------
# CUDA source – Fused Linear + Mish + Mish
# Uses tiling to handle large matrices where in_features might exceed 
# shared memory capacity or block limits.
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float mish(float x) {
    // x * tanh(softplus(x))
    // Use float approximation for speed: softplus(x) = log(1 + exp(x))
    float sp = logf(1.0f + expf(x));
    return x * tanhf(sp);
}

// Block size for tiling
#define TILE_W 32
#define TILE_H 32

__global__ void fused_linear_mish_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const int batch,
    const int in_features,
    const int out_features)
{
    // Shared memory tiling for input vector (row of x)
    __shared__ float s_x[TILE_W];
    __shared__ float s_w[TILE_W][TILE_H]; 

    const int b = blockIdx.y; 
    const int out_idx = blockIdx.x * TILE_H + threadIdx.x;
    
    float acc = (bias != nullptr && out_idx < out_features) ? bias[out_idx] : 0.0f;

    for (int k_tile = 0; k_tile < in_features; k_tile += TILE_W) {
        // Load one tile of x into shared memory
        if (k_tile + threadIdx.x < in_features) {
            s_x[threadIdx.x] = x[b * in_features + k_tile + threadIdx.x];
        } else {
            s_x[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Accumulate dot product
        if (out_idx < out_features) {
            #pragma unroll
            for (int k = 0; k < TILE_W; ++k) {
                acc += s_x[k] * weight[out_idx * in_features + k_tile + k];
            }
        }
        __syncthreads();
    }

    if (out_idx < out_features) {
        float y = mish(acc);
        y = mish(y);
        out[b * out_features + out_idx] = y;
    }
}

void fused_linear_mish(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out)
{
    const int batch = x.size(0);
    const int in_features = x.size(1);
    const int out_features = weight.size(0);

    // Grid: (ceil(out/TILE_H), batch)
    // Threads: TILE_W
    dim3 grid((out_features + TILE_H - 1) / TILE_H, batch);
    dim3 block(TILE_W);

    const float* b_ptr = (bias.numel() > 0) ? bias.data_ptr<float>() : nullptr;

    fused_linear_mish_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        b_ptr,
        out.data_ptr<float>(),
        batch,
        in_features,
        out_features
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_linear_mish(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_linear_mish, "Fused linear + 2xMish");
}
"""

fused_ext = load_inline(
    name='fused_op_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, linear_weight, linear_bias):
    device = x.device
    dtype = x.dtype
    batch = x.shape[0]
    out_features = linear_weight.shape[0]
    
    out = torch.empty((batch, out_features), device=device, dtype=dtype)
    bias = linear_bias if linear_bias is not None else torch.tensor([], device=device, dtype=dtype)
    
    fused_ext.fused_op(x, linear_weight, bias, out)
    return out
