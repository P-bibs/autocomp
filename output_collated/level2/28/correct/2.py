# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_145711/code_7.py
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --------------------------------------------------------------
# 1. CUDA kernel (Tiled GEMM + Bias)
# --------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void gemm_bias_kernel(const float* A, const float* B, const float* bias, float* C, 
                                 int M, int N, int K) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        if (row < M && (t * TILE_DIM + threadIdx.x) < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + (t * TILE_DIM + threadIdx.x)];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (t * TILE_DIM + threadIdx.y) < K)
            Bs[threadIdx.y][threadIdx.x] = B[col * K + (t * TILE_DIM + threadIdx.y)];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum + bias[col];
    }
}

void launch_gemm_bias(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& bias, torch::Tensor& C) {
    int M = A.size(0);
    int N = B.size(0);
    int K = A.size(1);
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    
    gemm_bias_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), 
                                      bias.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_gemm_bias(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& bias, torch::Tensor& C);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear", &launch_gemm_bias, "Fused Linear (GEMM+Bias)");
}
"""

fused_ext = load_inline(
    name='fused_linear_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x, y, *, bmm_weight, bmm_bias, instance_norm_running_mean, instance_norm_running_var,
    instance_norm_weight, instance_norm_bias, instance_norm_use_input_stats,
    instance_norm_momentum, instance_norm_eps,
):
    # Custom fused kernel
    out = torch.empty((x.shape[0], bmm_weight.shape[0]), device=x.device, dtype=x.dtype)
    fused_ext.fused_linear(x, bmm_weight, bmm_bias, out)
    
    # Instance norm
    out = F.instance_norm(
        out.unsqueeze(1).unsqueeze(1),
        instance_norm_running_mean, instance_norm_running_var,
        instance_norm_weight, instance_norm_bias,
        use_input_stats=instance_norm_use_input_stats,
        momentum=instance_norm_momentum, eps=instance_norm_eps
    ).squeeze(1).squeeze(1)
    
    return (out + y) * y
