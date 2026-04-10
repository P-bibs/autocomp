# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150602/code_1.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_op_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ W,
    const float* __restrict__ b,
    const float* __restrict__ y,
    float* __restrict__ out,
    int B, int N, int M, float eps) {

    // Each block handles one batch element
    int b_idx = blockIdx.x;
    if (b_idx >= B) return;

    // Shared memory for partial sums and normalization stats
    extern __shared__ float shared_mem[];
    float* s_linear_output = shared_mem;           // Size: M
    float* s_sum_partial = &shared_mem[M];         // Size: blockDim.x
    float* s_sum2_partial = &shared_mem[M + blockDim.x]; // Size: blockDim.x

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // 1. Compute Linear output: x @ W^T + b
    float linear_val = b[tid];
    if (tid < M) {
        for (int k = 0; k < N; ++k) {
            linear_val += x[b_idx * N + k] * W[tid * N + k];
        }
        s_linear_output[tid] = linear_val;
    } else {
        s_linear_output[tid] = 0.0f;
    }
    __syncthreads();

    // 2. Parallel reduction to compute sum and sum of squares for mean/variance
    float thread_sum = 0.0f;
    float thread_sum2 = 0.0f;
    
    for (int i = tid; i < M; i += block_size) {
        float val = s_linear_output[i];
        thread_sum += val;
        thread_sum2 += val * val;
    }
    s_sum_partial[tid] = thread_sum;
    s_sum2_partial[tid] = thread_sum2;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum_partial[tid] += s_sum_partial[tid + stride];
            s_sum2_partial[tid] += s_sum2_partial[tid + stride];
        }
        __syncthreads();
    }

    // 3. Compute mean and variance using Welford's online algorithm
    float mean = s_sum_partial[0] / M;
    float variance = (s_sum2_partial[0] / M) - (mean * mean);
    float inv_std = rsqrtf(variance + eps);

    // 4. Apply normalization, add y, multiply y
    if (tid < M) {
        float normalized = (s_linear_output[tid] - mean) * inv_std;
        float result = normalized + y[b_idx * M + tid];
        out[b_idx * M + tid] = result * y[b_idx * M + tid];
    }
}

void fused_op_forward(
    torch::Tensor x,
    torch::Tensor W,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor out) {
    
    int B = x.size(0);
    int N = x.size(1);
    int M = W.size(0);
    
    dim3 grid(B);
    dim3 block(min(1024, M));
    
    // Shared memory: M floats for linear output + 2 * block_size floats for reductions
    size_t shared_mem_size = (M + 2 * block.x) * sizeof(float);
    
    fused_op_forward_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        W.data_ptr<float>(),
        b.data_ptr<float>(),
        y.data_ptr<float>(),
        out.data_ptr<float>(),
        B, N, M, 1e-5f
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor x,
    torch::Tensor W,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Linear-InstanceNorm-Elementwise Operation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, y, *, bmm_weight, bmm_bias, **kwargs):
    out = torch.empty_like(y)
    fused_ext.fused_op(x, bmm_weight, bmm_bias, y, out)
    return out

batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features), torch.rand(batch_size, out_features)]
