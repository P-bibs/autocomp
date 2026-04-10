# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_151636/code_2.py
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

# --- CUDA Kernel Implementation ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void fused_op_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ norm_weight,
    const float* __restrict__ norm_bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ output,
    bool use_input_stats,
    float momentum,
    float eps,
    int batch_size,
    int in_features,
    int out_features
) {
    const int batch_idx = blockIdx.x;
    const int feature_start = threadIdx.x;
    const int feature_stride = blockDim.x;
    
    if (batch_idx >= batch_size) return;

    extern __shared__ float sdata[];
    float* shared_sum = sdata;
    float* shared_sum_sq = &sdata[out_features];

    // Shared memory for normalization stats
    __shared__ float mean_val, var_val;

    // Step 1: Linear Transformation (x @ W + b)
    for (int out_idx = feature_start; out_idx < out_features; out_idx += feature_stride) {
        float sum = (bias != nullptr) ? bias[out_idx] : 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += x[batch_idx * in_features + k] * weight[out_idx * in_features + k];
        }
        output[batch_idx * out_features + out_idx] = sum;
    }
    __syncthreads();

    // Step 2: Compute Instance Norm Statistics
    if (threadIdx.x == 0) {
        if (use_input_stats) {
            // Compute mean
            float local_sum = 0.0f;
            for (int i = 0; i < out_features; ++i) {
                local_sum += output[batch_idx * out_features + i];
            }
            mean_val = local_sum / out_features;

            // Compute variance
            float local_sum_sq = 0.0f;
            for (int i = 0; i < out_features; ++i) {
                float diff = output[batch_idx * out_features + i] - mean_val;
                local_sum_sq += diff * diff;
            }
            var_val = local_sum_sq / out_features;
        } else {
            // Use running stats
            mean_val = (running_mean != nullptr) ? running_mean[0] : 0.0f;
            var_val = (running_var != nullptr) ? running_var[0] : 1.0f;
        }
    }
    __syncthreads();

    // Step 3: Apply Instance Norm and Element-wise Ops
    for (int out_idx = feature_start; out_idx < out_features; out_idx += feature_stride) {
        float val = output[batch_idx * out_features + out_idx];
        float normalized = (val - mean_val) * rsqrtf(var_val + eps);
        if (norm_weight != nullptr) normalized *= norm_weight[0];
        if (norm_bias != nullptr) normalized += norm_bias[0];
        
        float y_val = y[batch_idx * out_features + out_idx];
        output[batch_idx * out_features + out_idx] = (normalized + y_val) * y_val;
    }
}

void fused_op_forward(
    const at::Tensor x,
    const at::Tensor y,
    const at::Tensor weight,
    const at::Tensor bias,
    const at::Tensor norm_weight,
    const at::Tensor norm_bias,
    const at::Tensor running_mean,
    const at::Tensor running_var,
    at::Tensor output,
    bool use_input_stats,
    float momentum,
    float eps,
    int batch_size,
    int in_features,
    int out_features
) {
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    const int threads_per_block = 256;
    const int blocks = batch_size;
    const int shared_mem_size = 2 * out_features * sizeof(float);

    fused_op_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        norm_weight.data_ptr<float>(),
        norm_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        use_input_stats,
        momentum,
        eps,
        batch_size,
        in_features,
        out_features
    );
    AT_CUDA_CHECK(cudaGetLastError());
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const at::Tensor x,
    const at::Tensor y,
    const at::Tensor weight,
    const at::Tensor bias,
    const at::Tensor norm_weight,
    const at::Tensor norm_bias,
    const at::Tensor running_mean,
    const at::Tensor running_var,
    at::Tensor output,
    bool use_input_stats,
    float momentum,
    float eps,
    int batch_size,
    int in_features,
    int out_features
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Linear + InstanceNorm + Add + Mul");
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
    batch_size, in_features = x.shape
    _, out_features = y.shape
    
    # Ensure tensors are contiguous
    x = x.contiguous()
    y = y.contiguous()
    bmm_weight = bmm_weight.contiguous()
    bmm_bias = bmm_bias.contiguous() if bmm_bias is not None else None
    instance_norm_weight = instance_norm_weight.contiguous() if instance_norm_weight is not None else None
    instance_norm_bias = instance_norm_bias.contiguous() if instance_norm_bias is not None else None
    instance_norm_running_mean = instance_norm_running_mean.contiguous() if instance_norm_running_mean is not None else None
    instance_norm_running_var = instance_norm_running_var.contiguous() if instance_norm_running_var is not None else None
    
    output = torch.empty_like(y)
    
    # Call the fused CUDA kernel
    fused_ext.fused_op(
        x, y,
        bmm_weight, bmm_bias,
        instance_norm_weight, instance_norm_bias,
        instance_norm_running_mean, instance_norm_running_var,
        output,
        instance_norm_use_input_stats,
        instance_norm_momentum,
        instance_norm_eps,
        batch_size, in_features, out_features
    )
    
    return output

batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features), torch.rand(batch_size, out_features)]
