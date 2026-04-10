# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_151636/code_5.py
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

# Tiled CUDA kernel to perform MatMul and fusion: (xW^T + b + y) * y
# Tiled approach improves memory reuse compared to the naive accumulation.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void fused_linear_post_kernel(const float* __restrict__ x, const float* __restrict__ w, 
                                        const float* __restrict__ b, const float* __restrict__ y, 
                                        float* __restrict__ out, int B, int N, int M) {
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
        float val = acc + b[col];
        float y_val = y[row * M + col];
        out[row * M + col] = (val + y_val) * y_val;
    }
}

void fused_op(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor y, torch::Tensor out) {
    int B = x.size(0);
    int N = x.size(1);
    int M = w.size(0);
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((M + TILE_SIZE - 1) / TILE_SIZE, (B + TILE_SIZE - 1) / TILE_SIZE);
    
    fused_linear_post_kernel<<<blocks, threads>>>(x.data_ptr<float>(), w.data_ptr<float>(), 
                                                 b.data_ptr<float>(), y.data_ptr<float>(), 
                                                 out.data_ptr<float>(), B, N, M);
}
"""

cpp_source = r"""
void fused_op(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor y, torch::Tensor out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused Linear and Elementwise operations");
}
"""

module = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, y, *, bmm_weight, bmm_bias, **kwargs):
    # Transpose weight because GEMM kernel expects (M, N) for w (where out=xW^T)
    # The original linear uses weight (out_features, in_features)
    out = torch.empty((x.size(0), bmm_weight.size(0)), device=x.device)
    module.fused_op(x, bmm_weight, bmm_bias, y, out)
    return out

batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    # Use float32 on GPU for high performance and kernel compatibility
    return [torch.rand(batch_size, in_features).cuda(), torch.rand(batch_size, out_features).cuda()]
