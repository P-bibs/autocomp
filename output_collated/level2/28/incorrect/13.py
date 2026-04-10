# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150223/code_0.py
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

# Define the fused CUDA kernel
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cmath>

__global__ void fused_op_forward_kernel(
    const float* x,
    const float* y,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    const float* norm_weight,
    const float* norm_bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    bool use_input_stats,
    float momentum,
    float eps
) {
    int batch_idx = blockIdx.x;
    int feature_idx = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Shared memory for reduction operations
    extern __shared__ float smem[];
    float* shared_data = smem;
    float* shared_mean = smem + out_features;
    float* shared_var = smem + out_features * 2;
    
    // Step 1: Perform linear operation (F.linear)
    float linear_result = (bias != nullptr) ? bias[feature_idx] : 0.0f;
    
    // Compute dot product for this output feature
    for (int i = 0; i < in_features; ++i) {
        linear_result += x[batch_idx * in_features + i] * weight[feature_idx * in_features + i];
    }
    
    shared_data[feature_idx] = linear_result;
    __syncthreads();
    
    // Step 2: Instance normalization - compute mean and variance per sample
    if (use_input_stats && feature_idx == 0) {
        // Compute mean
        float sum = 0.0f;
        for (int i = 0; i < out_features; ++i) {
            sum += shared_data[i];
        }
        float mean = sum / out_features;
        
        // Compute variance
        float sum_sq_diff = 0.0f;
        for (int i = 0; i < out_features; ++i) {
            float diff = shared_data[i] - mean;
            sum_sq_diff += diff * diff;
        }
        float var = sum_sq_diff / out_features;
        
        shared_mean[0] = mean;
        shared_var[0] = var;
    }
    
    __syncthreads();
    
    // Normalize using either computed stats or running stats
    float normalized_value;
    if (use_input_stats) {
        float mean = shared_mean[0];
        float var = shared_var[0];
        float inv_std = rsqrtf(var + eps);
        normalized_value = (linear_result - mean) * inv_std;
    } else {
        float inv_std = rsqrtf(running_var[feature_idx] + eps);
        normalized_value = (linear_result - running_mean[feature_idx]) * inv_std;
    }
    
    // Apply learnable parameters for instance norm
    float norm_result = normalized_value;
    if (norm_weight != nullptr) {
        norm_result *= norm_weight[feature_idx];
    }
    if (norm_bias != nullptr) {
        norm_result += norm_bias[feature_idx];
    }
    
    // Step 3 & 4: Element-wise addition and multiplication with y
    float add_result = norm_result + y[batch_idx * out_features + feature_idx];
    output[batch_idx * out_features + feature_idx] = add_result * y[batch_idx * out_features + feature_idx];
}

void fused_op_forward(
    const at::Tensor& x,
    const at::Tensor& y,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const c10::optional<at::Tensor>& norm_weight,
    const c10::optional<at::Tensor>& norm_bias,
    at::Tensor& output,
    bool use_input_stats,
    float momentum,
    float eps
) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = y.size(1);
    
    // Calculate grid and block dimensions
    dim3 grid(batch_size);
    dim3 block(out_features);
    
    // Shared memory size: data + mean + var
    size_t shared_mem_size = (out_features * 3) * sizeof(float);
    
    fused_op_forward_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        norm_weight.has_value() ? norm_weight.value().data_ptr<float>() : nullptr,
        norm_bias.has_value() ? norm_bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        use_input_stats,
        momentum,
        eps
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const at::Tensor& x,
    const at::Tensor& y,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const c10::optional<at::Tensor>& norm_weight,
    const c10::optional<at::Tensor>& norm_bias,
    at::Tensor& output,
    bool use_input_stats,
    float momentum,
    float eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op_forward", &fused_op_forward, "Fused operation forward pass");
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
    batch_size = x.shape[0]
    in_features = x.shape[1]
    out_features = y.shape[1]
    
    # Ensure tensors are on CUDA
    x = x.cuda()
    y = y.cuda()
    bmm_weight = bmm_weight.cuda()
    if bmm_bias is not None:
        bmm_bias = bmm_bias.cuda()
    instance_norm_running_mean = instance_norm_running_mean.cuda()
    instance_norm_running_var = instance_norm_running_var.cuda()
    if instance_norm_weight is not None:
        instance_norm_weight = instance_norm_weight.cuda()
    if instance_norm_bias is not None:
        instance_norm_bias = instance_norm_bias.cuda()
    
    # Allocate output tensor
    output = torch.empty_like(y, device='cuda')
    
    # Launch the fused kernel
    fused_ext.fused_op_forward(
        x.contiguous(),
        y.contiguous(),
        bmm_weight.contiguous(),
        bmm_bias,
        instance_norm_running_mean.contiguous(),
        instance_norm_running_var.contiguous(),
        instance_norm_weight,
        instance_norm_bias,
        output,
        instance_norm_use_input_stats,
        instance_norm_momentum,
        instance_norm_eps
    )
    
    return output

batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda'), torch.rand(batch_size, out_features, device='cuda')]
