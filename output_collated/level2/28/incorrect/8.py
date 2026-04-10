# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_145711/code_2.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused linear + instance_norm + add + multiply
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_linear_instancenorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ norm_weight,
    const float* __restrict__ norm_bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    float eps,
    bool use_input_stats
) {
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    if (batch_id >= batch_size) return;
    
    extern __shared__ float shared_mem[];
    float* shared_sum = shared_mem;
    float* shared_sum_sq = shared_mem + block_size;
    
    // Step 1: Compute linear transformation for all output features for this batch
    // Each thread computes multiple output features
    for (int out_id = tid; out_id < out_features; out_id += block_size) {
        float linear_result = bias[out_id];
        for (int i = 0; i < in_features; i++) {
            linear_result += x[batch_id * in_features + i] * weight[out_id * in_features + i];
        }
        shared_sum[tid] = linear_result;
        shared_sum_sq[tid] = linear_result * linear_result;
    }
    
    __syncthreads();
    
    // Reduction to compute mean and variance
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
            shared_sum_sq[tid] += shared_sum_sq[tid + s];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / out_features;
    float variance = shared_sum_sq[0] / out_features - mean * mean;
    
    // Process output features
    for (int out_id = tid; out_id < out_features; out_id += block_size) {
        // Recompute linear result for this output feature
        float linear_result = bias[out_id];
        for (int i = 0; i < in_features; i++) {
            linear_result += x[batch_id * in_features + i] * weight[out_id * in_features + i];
        }
        
        // Normalize
        float normalized = (linear_result - mean) / sqrtf(variance + eps);
        
        // Apply affine transformation
        float normalized_result = norm_weight[out_id] * normalized + norm_bias[out_id];
        
        // Add y and multiply by y
        float y_val = y[batch_id * out_features + out_id];
        float result = (normalized_result + y_val) * y_val;
        
        output[batch_id * out_features + out_id] = result;
    }
}

void fused_linear_instancenorm_forward(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor norm_weight,
    torch::Tensor norm_bias,
    torch::Tensor output,
    int batch_size,
    int in_features,
    int out_features,
    float eps,
    bool use_input_stats
) {
    // Use one block per batch element
    int threads_per_block = 512;
    int blocks = batch_size;
    size_t shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    fused_linear_instancenorm_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        norm_weight.data_ptr<float>(),
        norm_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        eps,
        use_input_stats
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_linear_instancenorm_forward(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor norm_weight,
    torch::Tensor norm_bias,
    torch::Tensor output,
    int batch_size,
    int in_features,
    int out_features,
    float eps,
    bool use_input_stats
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_instancenorm", &fused_linear_instancenorm_forward, 
          "Fused linear + instance norm + add + multiply operation");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False
)


def functional_model(
    x,
    y,
    *,
    bmm_weight,
    bmm_bias,
    instance_norm_running_mean,
    instance_norm_running_var,
    instance_norm_weight,
    instance_norm_bias,
    instance_norm_use_input_stats,
    instance_norm_momentum,
    instance_norm_eps,
):
    batch_size = x.shape[0]
    in_features = x.shape[1]
    out_features = bmm_weight.shape[0]
    
    output = torch.empty(batch_size, out_features, dtype=x.dtype, device=x.device)
    
    fused_ext.fused_linear_instancenorm(
        x.contiguous(),
        y.contiguous(),
        bmm_weight.contiguous(),
        bmm_bias.contiguous(),
        instance_norm_weight.contiguous(),
        instance_norm_bias.contiguous(),
        output,
        batch_size,
        in_features,
        out_features,
        instance_norm_eps,
        instance_norm_use_input_stats
    )
    
    return output


batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features), torch.rand(batch_size, out_features)]
