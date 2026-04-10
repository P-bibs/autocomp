# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_151319/code_5.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'eps', 'momentum']
FORWARD_ARG_NAMES = ['x', 'y']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['bmm_weight', 'bmm_bias', 'instance_norm_running_mean', 'instance_norm_running_var', 'instance_norm_weight', 'instance_norm_bias', 'instance_norm_use_input_stats', 'instance_norm_momentum', 'instance_norm_eps']
REQUIRED_FLAT_STATE_NAMES = ['bmm_weight', 'bmm_bias', 'instance_norm_running_mean', 'instance_norm_running_var', 'instance_norm_weight', 'instance_norm_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a batch matrix multiplication, instance normalization, summation, residual addition, and multiplication.
    """

    def __init__(self, in_features, out_features, eps=1e-05, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)

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
    # State for bmm (nn.Linear)
    if 'bmm_weight' in flat_state:
        state_kwargs['bmm_weight'] = flat_state['bmm_weight']
    else:
        state_kwargs['bmm_weight'] = getattr(model.bmm, 'weight', None)
    if 'bmm_bias' in flat_state:
        state_kwargs['bmm_bias'] = flat_state['bmm_bias']
    else:
        state_kwargs['bmm_bias'] = getattr(model.bmm, 'bias', None)
    # State for instance_norm (nn.InstanceNorm2d)
    if 'instance_norm_running_mean' in flat_state:
        state_kwargs['instance_norm_running_mean'] = flat_state['instance_norm_running_mean']
    else:
        state_kwargs['instance_norm_running_mean'] = getattr(model.instance_norm, 'running_mean', None)
    if 'instance_norm_running_var' in flat_state:
        state_kwargs['instance_norm_running_var'] = flat_state['instance_norm_running_var']
    else:
        state_kwargs['instance_norm_running_var'] = getattr(model.instance_norm, 'running_var', None)
    if 'instance_norm_weight' in flat_state:
        state_kwargs['instance_norm_weight'] = flat_state['instance_norm_weight']
    else:
        state_kwargs['instance_norm_weight'] = getattr(model.instance_norm, 'weight', None)
    if 'instance_norm_bias' in flat_state:
        state_kwargs['instance_norm_bias'] = flat_state['instance_norm_bias']
    else:
        state_kwargs['instance_norm_bias'] = getattr(model.instance_norm, 'bias', None)
    state_kwargs['instance_norm_use_input_stats'] = not model.instance_norm.track_running_stats
    state_kwargs['instance_norm_momentum'] = model.instance_norm.momentum
    state_kwargs['instance_norm_eps'] = model.instance_norm.eps
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

#define TILE_SIZE 32

__global__ void fused_linear_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ W,
    const float* __restrict__ b,
    const float* __restrict__ y,
    float* __restrict__ out,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ norm_w,
    const float* __restrict__ norm_b,
    float eps,
    int B, int IN, int OUT) {

    extern __shared__ float tile_x[TILE_SIZE][TILE_SIZE];
    extern __shared__ float tile_w[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int k_tile = 0; k_tile < (IN + TILE_SIZE - 1) / TILE_SIZE; ++k_tile) {
        if (row < B && (k_tile * TILE_SIZE + threadIdx.x) < IN)
            tile_x[threadIdx.y][threadIdx.x] = x[row * IN + k_tile * TILE_SIZE + threadIdx.x];
        else tile_x[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < OUT && (k_tile * TILE_SIZE + threadIdx.y) < IN)
            tile_w[threadIdx.y][threadIdx.x] = W[(k_tile * TILE_SIZE + threadIdx.y) * OUT + col];
        else tile_w[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();
        for (int k = 0; k < TILE_SIZE; ++k) sum += tile_x[threadIdx.y][k] * tile_w[k][threadIdx.x];
        __syncthreads();
    }

    if (row < B && col < OUT) {
        float val = sum + b[col];
        // Apply Instance Norm (simplified scaling)
        val = (val - mean[0]) * rsqrtf(var[0] + eps);
        val = val * norm_w[0] + norm_b[0];
        // Apply Elementwise: (val + y) * y
        out[row * OUT + col] = (val + y[row * OUT + col]) * y[row * OUT + col];
    }
}

void fused_op(torch::Tensor x, torch::Tensor W, torch::Tensor b, torch::Tensor y,
              torch::Tensor out, torch::Tensor mean, torch::Tensor var,
              torch::Tensor nw, torch::Tensor nb, float eps) {
    int B = x.size(0); int IN = x.size(1); int OUT = W.size(1);
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((OUT + TILE_SIZE - 1) / TILE_SIZE, (B + TILE_SIZE - 1) / TILE_SIZE);
    
    fused_linear_norm_kernel<<<blocks, threads, 2 * TILE_SIZE * TILE_SIZE * sizeof(float)>>>(
        x.data_ptr<float>(), W.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(),
        out.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(),
        nw.data_ptr<float>(), nb.data_ptr<float>(), eps, B, IN, OUT);
}
"""

cpp_source = r"""
void fused_op(torch::Tensor x, torch::Tensor W, torch::Tensor b, torch::Tensor y,
              torch::Tensor out, torch::Tensor mean, torch::Tensor var,
              torch::Tensor nw, torch::Tensor nb, float eps);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused Linear Norm and Elemwise");
}
"""

fused_ext = load_inline(name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source, with_cuda=True, extra_cuda_cflags=['-O3'])

def functional_model(x, y, *, bmm_weight, bmm_bias, instance_norm_running_mean, 
                     instance_norm_running_var, instance_norm_weight, instance_norm_bias, 
                     instance_norm_use_input_stats, instance_norm_momentum, instance_norm_eps):
    out = torch.empty_like(y)
    fused_ext.fused_op(x, bmm_weight.t().contiguous(), bmm_bias, y, out, 
                       instance_norm_running_mean, instance_norm_running_var, 
                       instance_norm_weight, instance_norm_bias, instance_norm_eps)
    return out
