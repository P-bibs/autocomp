# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150602/code_5.py
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

# The kernel performs fused Linear + Instance Norm + Element-wise operations.
# We use a two-pass reduction approach for Instance Norm (mean/var) to avoid sync issues.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fused_kernel(
    const float* __restrict__ x, const float* __restrict__ W, const float* __restrict__ b,
    const float* __restrict__ y, float* __restrict__ out,
    int B, int N, int M, float eps) {

    extern __shared__ float s_mem[]; 
    float* row_vals = s_mem; // Size M

    int b_idx = blockIdx.x;
    
    // 1. Compute Linear: row_vals[m] = sum(x[n] * W[m, n]) + b[m]
    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        float acc = b[m];
        for (int n = 0; n < N; ++n) {
            acc += x[b_idx * N + n] * W[m * N + n];
        }
        row_vals[m] = acc;
    }
    __syncthreads();

    // 2. Compute Mean
    float sum = 0;
    for (int i = threadIdx.x; i < M; i += blockDim.x) sum += row_vals[i];
    
    // Parallel reduction for mean
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    float mean = (__shfl_sync(0xffffffff, sum, 0)) / M;
    __syncthreads();

    // 3. Compute Var
    float sq_diff_sum = 0;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        float diff = row_vals[i] - mean;
        sq_diff_sum += diff * diff;
    }
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        sq_diff_sum += __shfl_down_sync(0xffffffff, sq_diff_sum, offset);
    }
    float var = (__shfl_sync(0xffffffff, sq_diff_sum, 0)) / M;
    float inv_std = rsqrtf(var + eps);

    // 4. Apply Norm and element-wise ops
    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        float norm_val = (row_vals[m] - mean) * inv_std;
        float y_val = y[b_idx * M + m];
        out[b_idx * M + m] = (norm_val + y_val) * y_val;
    }
}

void fused_op_cuda(torch::Tensor x, torch::Tensor W, torch::Tensor b, torch::Tensor y, torch::Tensor out) {
    int B = x.size(0);
    int N = x.size(1);
    int M = W.size(0);
    
    // Launch one block per batch item
    int threads = 512;
    size_t shared_mem = M * sizeof(float);
    fused_kernel<<<B, threads, shared_mem>>>(
        x.data_ptr<float>(), W.data_ptr<float>(), b.data_ptr<float>(),
        y.data_ptr<float>(), out.data_ptr<float>(), B, N, M, 1e-5f
    );
}
"""

cpp_source = r"""
void fused_op_cuda(torch::Tensor x, torch::Tensor W, torch::Tensor b, torch::Tensor y, torch::Tensor out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_cuda, "Fused Linear Norm Arithmetic kernel");
}
"""

# Compile the extension
fused_module = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, y, *, bmm_weight, bmm_bias, **kwargs):
    # Prepare output tensor
    out = torch.empty_like(y)
    # Ensure inputs are contiguous float32
    fused_module.fused_op(
        x.contiguous(), 
        bmm_weight.contiguous(), 
        bmm_bias.contiguous(), 
        y.contiguous(), 
        out
    )
    return out
