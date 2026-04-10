# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150941/code_0.py
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

# CUDA kernel implementation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Atomic.cuh>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;
    
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

__global__ void fused_functional_model_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ bmm_weight,
    const float* __restrict__ bmm_bias,
    const float* __restrict__ instance_norm_running_mean,
    const float* __restrict__ instance_norm_running_var,
    const float* __restrict__ instance_norm_weight,
    const float* __restrict__ instance_norm_bias,
    bool instance_norm_use_input_stats,
    float instance_norm_momentum,
    float instance_norm_eps,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int out_idx = blockIdx.y * blockDim.x + tid;
    
    if (batch_idx >= batch_size) return;
    
    // Registers for computation
    float linear_result = 0.0f;
    
    // Linear transformation - compute one output element
    if (out_idx < out_features) {
        for (int k = 0; k < in_features; k++) {
            linear_result += x[batch_idx * in_features + k] * bmm_weight[out_idx * in_features + k];
        }
        linear_result += bmm_bias[out_idx];
    }
    
    // Instance normalization
    if (out_idx < out_features) {
        float mean, var;
        if (instance_norm_use_input_stats) {
            // Compute mean using cooperative reduction
            __shared__ float shared_mean;
            float thread_mean = linear_result;
            thread_mean = blockReduceSum(thread_mean);
            if (tid == 0) {
                shared_mean = thread_mean / (float)out_features;
            }
            __syncthreads();
            mean = shared_mean;
            
            // Compute variance using cooperative reduction
            __shared__ float shared_var;
            float thread_var = (linear_result - mean) * (linear_result - mean);
            thread_var = blockReduceSum(thread_var);
            if (tid == 0) {
                shared_var = thread_var / (float)out_features;
            }
            __syncthreads();
            var = shared_var;
        } else {
            // Use provided running stats
            mean = instance_norm_running_mean[out_idx];
            var = instance_norm_running_var[out_idx];
        }
        
        float normalized = (linear_result - mean) * rsqrtf(var + instance_norm_eps);
        float scaled_shifted = normalized * instance_norm_weight[out_idx] + instance_norm_bias[out_idx];
        
        // Final operations
        float y_val = y[batch_idx * out_features + out_idx];
        output[batch_idx * out_features + out_idx] = (scaled_shifted + y_val) * y_val;
    }
}

void fused_functional_model_forward(
    const at::Tensor& x,
    const at::Tensor& y,
    const at::Tensor& bmm_weight,
    const at::Tensor& bmm_bias,
    const at::Tensor& instance_norm_running_mean,
    const at::Tensor& instance_norm_running_var,
    const at::Tensor& instance_norm_weight,
    const at::Tensor& instance_norm_bias,
    bool instance_norm_use_input_stats,
    float instance_norm_momentum,
    float instance_norm_eps,
    at::Tensor& output
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = y.size(1);
    
    // Grid and block dimensions
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(batch_size, (out_features + block_size.x - 1) / block_size.x);
    
    fused_functional_model_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        bmm_weight.data_ptr<float>(),
        bmm_bias.data_ptr<float>(),
        instance_norm_running_mean.data_ptr<float>(),
        instance_norm_running_var.data_ptr<float>(),
        instance_norm_weight.data_ptr<float>(),
        instance_norm_bias.data_ptr<float>(),
        instance_norm_use_input_stats,
        instance_norm_momentum,
        instance_norm_eps,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ interface bindings
cpp_source = r"""
#include <torch/extension.h>

void fused_functional_model_forward(
    const at::Tensor& x,
    const at::Tensor& y,
    const at::Tensor& bmm_weight,
    const at::Tensor& bmm_bias,
    const at::Tensor& instance_norm_running_mean,
    const at::Tensor& instance_norm_running_var,
    const at::Tensor& instance_norm_weight,
    const at::Tensor& instance_norm_bias,
    bool instance_norm_use_input_stats,
    float instance_norm_momentum,
    float instance_norm_eps,
    at::Tensor& output
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_functional_model_forward", &fused_functional_model_forward, "Fused functional model forward pass");
}
"""

# Compile the CUDA extension
fused_ext = load_inline(
    name='fused_functional_model_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=compute_75', '-code=sm_75'],
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
    # Create output tensor with the same shape as expected
    output = torch.empty_like(y)
    
    # Call the fused CUDA kernel
    fused_ext.fused_functional_model_forward(
        x, y,
        bmm_weight, bmm_bias,
        instance_norm_running_mean, instance_norm_running_var,
        instance_norm_weight, instance_norm_bias,
        instance_norm_use_input_stats,
        instance_norm_momentum,
        instance_norm_eps,
        output
    )
    
    return output

batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda(), torch.rand(batch_size, out_features).cuda()]

# Move all tensors to GPU for performance
def setup_inputs():
    inputs = get_inputs()
    return [inp.cuda() for inp in inputs]
