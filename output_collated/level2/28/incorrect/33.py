# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_151319/code_7.py
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

# ----------------------------------------------------------------------
# CUDA Source Code
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Optimized Tiled GEMM with Bias addition
// M = batch_size, N = out_features, K = in_features
__global__ void gemm_bias_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int BLOCK_SIZE = 32;
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float acc = 0.0f;
    for (int k = 0; k < K; k += BLOCK_SIZE) {
        if (row < M && (k + threadIdx.x) < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + k + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (k + threadIdx.y) < K)
            Bs[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i)
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc + bias[col];
    }
}

// Fused kernel: Instance Norm (Statistics + Norm + Scale + Bias) + Add(y) + Mul(y)
__global__ void fused_norm_op_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ run_mean,
    const float* __restrict__ run_var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    bool use_stats, float eps, int M, int N)
{
    int row = blockIdx.x;
    if (row >= M) return;

    // Use fast shared memory based reduction if using input stats
    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sq = &sdata[blockDim.x];

    float sum = 0.0f;
    float sq_sum = 0.0f;
    
    if (use_stats) {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float val = x[row * N + i];
            sum += val;
            sq_sum += val * val;
        }
        s_sum[threadIdx.x] = sum;
        s_sq[threadIdx.x] = sq_sum;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
                s_sq[threadIdx.x] += s_sq[threadIdx.x + s];
            }
            __syncthreads();
        }
        sum = s_sum[0] / N;
        sq_sum = rsqrtf((s_sq[0] / N) - (sum * sum) + eps);
    }

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float val = x[row * N + i];
        float m = use_stats ? sum : run_mean[i];
        float inv_std = use_stats ? sq_sum : rsqrtf(run_var[i] + eps);
        
        float norm = (val - m) * inv_std;
        float scaled = norm * weight[i] + bias[i];
        
        float y_val = y[row * N + i];
        out[row * N + i] = (scaled + y_val) * y_val;
    }
}

void launch_fused(torch::Tensor x, torch::Tensor y, torch::Tensor w, torch::Tensor b, 
                  torch::Tensor rm, torch::Tensor rv, torch::Tensor rw, torch::Tensor rb, 
                  torch::Tensor out, bool use_stats, float eps) {
    int M = x.size(0);
    int N = w.size(1);
    int K = x.size(1);
    
    // Pass 1: GEMM
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    dim3 block(32, 32);
    gemm_bias_kernel<<<grid, block>>>(x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), M, N, K);

    // Pass 2: Fused Norm/Add/Mul
    int threads = 256;
    fused_norm_op_kernel<<<M, threads, 2 * threads * sizeof(float)>>>(
        out.data_ptr<float>(), y.data_ptr<float>(), rm.data_ptr<float>(), rv.data_ptr<float>(), 
        rw.data_ptr<float>(), rb.data_ptr<float>(), out.data_ptr<float>(), use_stats, eps, M, N);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_fused(torch::Tensor x, torch::Tensor y, torch::Tensor w, torch::Tensor b, 
                  torch::Tensor rm, torch::Tensor rv, torch::Tensor rw, torch::Tensor rb, 
                  torch::Tensor out, bool use_stats, float eps);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_fused, "Fused GEMM + Norm + Add + Mul");
}
"""

fused_ext = load_inline(name='fused_ops', cpp_sources=cpp_source, cuda_sources=cuda_source, extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, y, *, bmm_weight, bmm_bias, instance_norm_running_mean, instance_norm_running_var, 
                     instance_norm_weight, instance_norm_bias, instance_norm_use_input_stats, 
                     instance_norm_momentum, instance_norm_eps):
    # Ensure contiguous
    x, y = x.contiguous().cuda(), y.contiguous().cuda()
    w = bmm_weight.t().contiguous().cuda()
    b = bmm_bias.contiguous().cuda()
    rm = instance_norm_running_mean.contiguous().cuda()
    rv = instance_norm_running_var.contiguous().cuda()
    rw = instance_norm_weight.contiguous().cuda()
    rb = instance_norm_bias.contiguous().cuda()
    out = torch.empty_like(x) # Placeholder size for output
    
    fused_ext.fused_op(x, y, w, b, rm, rv, rw, rb, out, instance_norm_use_input_stats, instance_norm_eps)
    return out
