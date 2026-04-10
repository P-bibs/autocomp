# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150602/code_0.py
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

# Fused CUDA kernel implementation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_op_kernel(
    const float* x,
    const float* y,
    const float* bmm_weight,
    const float* bmm_bias,
    const float* instance_norm_running_mean,
    const float* instance_norm_running_var,
    const float* instance_norm_weight,
    const float* instance_norm_bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    bool instance_norm_use_input_stats,
    float instance_norm_momentum,
    float instance_norm_eps
) {
    int batch_idx = blockIdx.x;
    int feat_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Shared memory for reduction operations
    extern __shared__ float sdata[];
    
    // Linear operation: compute x @ weight.T + bias for this batch element
    float linear_result = 0.0f;
    if (bmm_bias != nullptr && feat_idx < out_features) {
        linear_result = bmm_bias[feat_idx];
    }
    
    // Compute dot product for linear transformation
    for (int i = 0; i < in_features; ++i) {
        if (feat_idx < out_features) {
            linear_result += x[batch_idx * in_features + i] * bmm_weight[feat_idx * in_features + i];
        }
    }
    
    // Instance normalization computation
    // First, compute mean and variance across features for this batch element
    __shared__ float mean, var;
    
    if (instance_norm_use_input_stats) {
        // Calculate mean
        sdata[threadIdx.x] = (feat_idx < out_features) ? linear_result : 0.0f;
        __syncthreads();
        
        // Reduction to compute mean
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s && (blockIdx.y * blockDim.x + threadIdx.x + s) < out_features) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }
            __syncthreads();
        }
        
        if (threadIdx.x == 0) {
            mean = sdata[0] / out_features;
        }
        __syncthreads();
        
        // Calculate variance
        sdata[threadIdx.x] = (feat_idx < out_features) ? 
            (linear_result - mean) * (linear_result - mean) : 0.0f;
        __syncthreads();
        
        // Reduction to compute variance
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s && (blockIdx.y * blockDim.x + threadIdx.x + s) < out_features) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }
            __syncthreads();
        }
        
        if (threadIdx.x == 0) {
            var = sdata[0] / out_features;
        }
        __syncthreads();
    } else {
        // Use running statistics
        if (threadIdx.x == 0) {
            // For simplicity, we're using the feature index here, but in a real implementation
            // you might want to handle this differently based on how running stats are stored
            mean = (feat_idx < out_features) ? instance_norm_running_mean[feat_idx % out_features] : 0.0f;
            var = (feat_idx < out_features) ? instance_norm_running_var[feat_idx % out_features] : 1.0f;
        }
        __syncthreads();
    }
    
    // Normalize, scale, and shift
    if (feat_idx < out_features) {
        float normalized = (linear_result - mean) / sqrtf(var + instance_norm_eps);
        float norm_result = normalized * instance_norm_weight[feat_idx] + instance_norm_bias[feat_idx];
        
        // Add and multiply with y
        float add_result = norm_result + y[batch_idx * out_features + feat_idx];
        output[batch_idx * out_features + feat_idx] = add_result * y[batch_idx * out_features + feat_idx];
    }
}

void fused_op_forward(
    const torch::Tensor x,
    const torch::Tensor y,
    const torch::Tensor bmm_weight,
    const torch::Tensor bmm_bias,
    const torch::Tensor instance_norm_running_mean,
    const torch::Tensor instance_norm_running_var,
    const torch::Tensor instance_norm_weight,
    const torch::Tensor instance_norm_bias,
    torch::Tensor output,
    bool instance_norm_use_input_stats,
    float instance_norm_momentum,
    float instance_norm_eps
) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = bmm_weight.size(0);
    
    // Grid and block dimensions
    int threads_per_block = min(256, ((out_features + 31) / 32) * 32);
    int blocks_per_grid_y = (out_features + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(batch_size, blocks_per_grid_y);
    dim3 block(threads_per_block);
    int shared_mem_size = threads_per_block * sizeof(float);
    
    fused_op_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        bmm_weight.data_ptr<float>(),
        bmm_bias.data_ptr<float>(),
        instance_norm_running_mean.data_ptr<float>(),
        instance_norm_running_var.data_ptr<float>(),
        instance_norm_weight.data_ptr<float>(),
        instance_norm_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        instance_norm_use_input_stats,
        instance_norm_momentum,
        instance_norm_eps
    );
}
"""

# C++ bindings
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const torch::Tensor x,
    const torch::Tensor y,
    const torch::Tensor bmm_weight,
    const torch::Tensor bmm_bias,
    const torch::Tensor instance_norm_running_mean,
    const torch::Tensor instance_norm_running_var,
    const torch::Tensor instance_norm_weight,
    const torch::Tensor instance_norm_bias,
    torch::Tensor output,
    bool instance_norm_use_input_stats,
    float instance_norm_momentum,
    float instance_norm_eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused operation forward pass");
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
    # Create output tensor
    output = torch.empty_like(y)
    
    # Call fused operation
    fused_ext.fused_op(
        x, y,
        bmm_weight, bmm_bias,
        instance_norm_running_mean, instance_norm_running_var,
        instance_norm_weight, instance_norm_bias,
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
    return [torch.rand(batch_size, in_features), torch.rand(batch_size, out_features)]
