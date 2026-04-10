# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_145329/code_2.py
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

__device__ inline float atomicMaxFloat(float* address, float val) {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void fused_op_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ norm_weight,
    const float* __restrict__ norm_bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_features,
    const int out_features,
    const float eps
) {
    const int batch_idx = blockIdx.x;
    const int out_feat_start = blockIdx.y * blockDim.x;
    const int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const int out_feat_idx = out_feat_start + tid;
    
    // Shared memory for reduction
    extern __shared__ float sdata[];
    float* s_mean = sdata;
    float* s_var = &sdata[blockDim.x];
    
    // Step 1: Linear Transformation (y = x @ W^T + b)
    float linear_result = 0.0f;
    if (out_feat_idx < out_features) {
        for (int i = 0; i < in_features; ++i) {
            linear_result += x[batch_idx * in_features + i] * weight[out_feat_idx * in_features + i];
        }
        linear_result += bias[out_feat_idx];
    }
    
    // Step 2: Instance Norm Statistics Computation
    // Since we're doing instance norm on a single feature vector per sample,
    // we compute mean and variance across the feature dimension
    
    // Initialize shared memory
    if (tid < blockDim.x) {
        s_mean[tid] = 0.0f;
        s_var[tid] = 0.0f;
    }
    __syncthreads();
    
    // Each thread loads one element and contributes to sum
    if (out_feat_idx < out_features) {
        s_mean[tid] = linear_result;
    }
    __syncthreads();
    
    // Reduction for mean
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (out_feat_start + tid + s) < out_features) {
            s_mean[tid] += s_mean[tid + s];
        }
        __syncthreads();
    }
    
    float mean = s_mean[0] / out_features;
    __syncthreads();
    
    // Compute variance
    if (out_feat_idx < out_features) {
        s_var[tid] = (linear_result - mean) * (linear_result - mean);
    }
    __syncthreads();
    
    // Reduction for variance
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (out_feat_start + tid + s) < out_features) {
            s_var[tid] += s_var[tid + s];
        }
        __syncthreads();
    }
    
    float variance = s_var[0] / out_features;
    __syncthreads();
    
    // Step 3: Normalize, scale, shift, add y, multiply by y
    if (out_feat_idx < out_features) {
        float normalized = (linear_result - mean) / sqrtf(variance + eps);
        float scaled_shifted = normalized * norm_weight[0] + norm_bias[0]; // Assuming scalar weights for simplicity
        float added = scaled_shifted + y[batch_idx * out_features + out_feat_idx];
        output[batch_idx * out_features + out_feat_idx] = added * y[batch_idx * out_features + out_feat_idx];
    }
}

void fused_op_forward(
    const torch::Tensor x,
    const torch::Tensor y,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const torch::Tensor norm_weight,
    const torch::Tensor norm_bias,
    torch::Tensor output,
    const float eps
) {
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    
    const int batch_size = x.size(0);
    const int in_features = x.size(1);
    const int out_features = y.size(1);
    
    // Kernel launch configuration
    const int threads_per_block = 256;
    const int blocks_per_feature = (out_features + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(batch_size, blocks_per_feature);
    dim3 block(threads_per_block);
    
    const int shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    fused_op_kernel<<<grid, block, shared_mem_size>>>(
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
        eps
    );
    
    cudaDeviceSynchronize(); // Ensure kernel completion
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const torch::Tensor x,
    const torch::Tensor y,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const torch::Tensor norm_weight,
    const torch::Tensor norm_bias,
    torch::Tensor output,
    const float eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused operation: Linear + InstanceNorm + Elementwise Add/Mul");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op_ext',
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
    
    # Call the fused CUDA kernel
    fused_ext.fused_op(
        x, y, 
        bmm_weight, bmm_bias,
        instance_norm_weight, instance_norm_bias,
        output,
        instance_norm_eps
    )
    
    return output

batch_size = 1024  # Increased batch size
in_features = 8192  # Increased input features
out_features = 8192  # Increased output features

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features), torch.rand(batch_size, out_features)]
