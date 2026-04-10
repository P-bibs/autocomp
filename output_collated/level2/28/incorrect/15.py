# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150223/code_2.py
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

# CUDA kernel source code
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_op_forward_kernel(
    const float* __restrict__ x,           // [batch_size, in_features]
    const float* __restrict__ y,           // [batch_size, out_features]
    const float* __restrict__ weight,      // [out_features, in_features]
    const float* __restrict__ bias,        // [out_features]
    const float* __restrict__ norm_weight, // [out_features]
    const float* __restrict__ norm_bias,   // [out_features]
    const float eps,
    float* __restrict__ output,            // [batch_size, out_features]
    int batch_size,
    int in_features,
    int out_features
) {
    // Each block processes one sample
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    // Thread index within block
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Pointers to current sample data
    const float* x_sample = x + batch_idx * in_features;
    const float* y_sample = y + batch_idx * out_features;
    float* out_sample = output + batch_idx * out_features;
    
    // Shared memory for reduction operations
    extern __shared__ float sdata[];
    float* s_vals = sdata;                          // Store linear outputs
    float* s_sum = sdata + out_features;            // Partial sums for mean
    float* s_sum_sq = sdata + 2 * out_features;     // Partial sums for variance
    
    // Step 1: Linear transformation (each thread computes multiple outputs if needed)
    for (int i = tid; i < out_features; i += block_size) {
        float val = (bias != nullptr) ? bias[i] : 0.0f;
        const float* w_row = weight + i * in_features;
        
        // Compute dot product
        for (int j = 0; j < in_features; ++j) {
            val += w_row[j] * x_sample[j];
        }
        s_vals[i] = val;
    }
    __syncthreads();
    
    // Step 2: Instance normalization - compute mean
    float sum = 0.0f;
    for (int i = tid; i < out_features; i += block_size) {
        sum += s_vals[i];
    }
    s_sum[tid] = sum;
    __syncthreads();
    
    // Reduce within block to compute mean
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    float mean = s_sum[0] / out_features;
    __syncthreads();
    
    // Step 3: Instance normalization - compute variance
    float sum_sq = 0.0f;
    for (int i = tid; i < out_features; i += block_size) {
        float diff = s_vals[i] - mean;
        sum_sq += diff * diff;
    }
    s_sum_sq[tid] = sum_sq;
    __syncthreads();
    
    // Reduce within block to compute variance
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }
    float variance = s_sum_sq[0] / out_features;
    float inv_std = rsqrtf(variance + eps);  // Use fast reciprocal square root
    __syncthreads();
    
    // Step 4: Normalize, scale/bias, and apply element-wise operations
    for (int i = tid; i < out_features; i += block_size) {
        // Normalize
        float normalized = (s_vals[i] - mean) * inv_std;
        
        // Apply instance norm weight and bias
        float scaled = normalized;
        if (norm_weight != nullptr) {
            scaled *= norm_weight[i];
        }
        if (norm_bias != nullptr) {
            scaled += norm_bias[i];
        }
        
        // Element-wise operations: (x + y) * y
        float result = (scaled + y_sample[i]) * y_sample[i];
        out_sample[i] = result;
    }
}

void fused_op_forward(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor norm_weight,
    torch::Tensor norm_bias,
    float eps,
    torch::Tensor output
) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);
    
    // Launch configuration
    int threads_per_block = min(512, ((out_features + 31) / 32) * 32); // Round to nearest multiple of 32
    int blocks = batch_size;
    
    // Shared memory size: values + partial sums for mean + partial sums for variance
    int shared_mem_size = 3 * out_features * sizeof(float);
    
    fused_op_forward_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        norm_weight.data_ptr<float>(),
        norm_bias.data_ptr<float>(),
        eps,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor norm_weight,
    torch::Tensor norm_bias,
    float eps,
    torch::Tensor output
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused linear + instance norm + element-wise operations");
}
"""

# Compile the extension
try:
    fused_ext = load_inline(
        name='fused_op',
        cpp_sources=cpp_source,
        cuda_sources=cuda_kernel,
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        with_cuda=True
    )
except Exception as e:
    print(f"Compilation note: {e}")
    fused_ext = None

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
    """
    Fused implementation of linear + instance norm + element-wise operations.
    """
    batch_size = x.size(0)
    out_features = bmm_weight.size(0)
    
    # Allocate output tensor
    output = torch.zeros(batch_size, out_features, dtype=x.dtype, device=x.device)
    
    # Call fused CUDA kernel
    fused_ext.fused_op(
        x.contiguous(),
        y.contiguous(),
        bmm_weight.contiguous(),
        bmm_bias if bmm_bias is not None else torch.empty(0, dtype=x.dtype, device=x.device),
        instance_norm_weight if instance_norm_weight is not None else torch.empty(0, dtype=x.dtype, device=x.device),
        instance_norm_bias if instance_norm_bias is not None else torch.empty(0, dtype=x.dtype, device=x.device),
        instance_norm_eps,
        output
    )
    
    return output


def get_init_inputs():
    batch_size = 1024
    in_features = 8192
    out_features = 8192
    return [in_features, out_features]


def get_inputs():
    batch_size = 1024
    in_features = 8192
    out_features = 8192
    return [torch.rand(batch_size, in_features), torch.rand(batch_size, out_features)]
