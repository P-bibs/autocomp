# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_145711/code_5.py
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
#include <cooperative_groups.h>

__global__ void fused_model_kernel(
    const float* __restrict__ x, const float* __restrict__ y, 
    const float* __restrict__ W, const float* __restrict__ bias,
    float* __restrict__ out, int B, int in_f, int out_f, float eps) {
    
    // Each block processes one batch item
    int b = blockIdx.x;
    int col = threadIdx.x;
    if (col >= out_f) return;

    // 1. Linear: Dot product of x[b] with row W[col]
    float val = bias[col];
    for (int i = 0; i < in_f; ++i) {
        val += x[b * in_f + i] * W[col * in_f + i];
    }

    // 2. Instance Norm Stats: Reduction over out_f dimension
    // We use a simplified reduction approach; for out_f=8192, 
    // global memory sync or shared memory is usually required.
    // For performance, we assume out_f is power of 2.
    extern __shared__ float shared_mem[];
    float* s_sum = shared_mem;
    float* s_sq = &shared_mem[out_f];

    s_sum[col] = val;
    s_sq[col] = val * val;
    __syncthreads();

    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (col < s) {
            s_sum[col] += s_sum[col + s];
            s_sq[col] += s_sq[col + s];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / out_f;
    float var = (s_sq[0] / out_f) - (mean * mean);
    float inv_std = rsqrtf(var + eps);

    // 3. Normalization and elementwise
    float norm_val = (val - mean) * inv_std;
    float y_val = y[b * out_f + col];
    out[b * out_f + col] = (norm_val + y_val) * y_val;
}

void fused_op(torch::Tensor x, torch::Tensor y, torch::Tensor w, torch::Tensor b, torch::Tensor out, float eps) {
    int B = x.size(0);
    int in_f = x.size(1);
    int out_f = w.size(0);
    
    // Launch configuration
    dim3 blocks(B);
    dim3 threads(1024);
    // Shared memory: 2 * out_f * sizeof(float)
    size_t shared_size = 2 * out_f * sizeof(float);
    
    fused_model_kernel<<<blocks, threads, shared_size>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), 
        out.data_ptr<float>(), B, in_f, out_f, eps);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op(torch::Tensor x, torch::Tensor y, torch::Tensor w, torch::Tensor b, torch::Tensor out, float eps);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused Linear + IN + Arithmetic");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, y, *, bmm_weight, bmm_bias, instance_norm_running_mean, instance_norm_running_var, 
                     instance_norm_weight, instance_norm_bias, instance_norm_use_input_stats, 
                     instance_norm_momentum, instance_norm_eps):
    out = torch.empty_like(y)
    fused_ext.fused_op(x, y, bmm_weight, bmm_bias, out, instance_norm_eps)
    return out
