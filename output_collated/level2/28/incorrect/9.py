# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_145711/code_1.py
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
#include <cuda_fp16.h>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void fused_op_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ norm_weight,
    const float* __restrict__ norm_bias,
    float* __restrict__ out,
    int batch_size,
    int in_features,
    int out_features,
    float eps
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    if (batch_idx >= batch_size) return;
    
    // Shared memory for reductions
    extern __shared__ float shared_mem[];
    float* shared_sum = shared_mem;
    float* shared_sq_sum = shared_mem + WARP_SIZE;
    
    // Linear transformation: out = x @ weight.T + bias
    float linear_val = 0.0f;
    if (tid < out_features) {
        linear_val = bias[tid];
        for (int i = 0; i < in_features; ++i) {
            linear_val += x[batch_idx * in_features + i] * weight[tid * in_features + i];
        }
    }
    
    // Instance normalization statistics computation
    float sum = linear_val;
    float sq_sum = linear_val * linear_val;
    
    // Warp-level reduction
    sum = warp_reduce_sum(sum);
    sq_sum = warp_reduce_sum(sq_sum);
    
    // Write to shared memory
    if (lane_id == 0) {
        shared_sum[warp_id] = sum;
        shared_sq_sum[warp_id] = sq_sum;
    }
    __syncthreads();
    
    // Final reduction using first warp
    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? shared_sum[lane_id] : 0.0f;
        sq_sum = (lane_id < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? shared_sq_sum[lane_id] : 0.0f;
        
        sum = warp_reduce_sum(sum);
        sq_sum = warp_reduce_sum(sq_sum);
        
        if (lane_id == 0) {
            shared_sum[0] = sum;
            shared_sq_sum[0] = sq_sum;
        }
    }
    __syncthreads();
    
    // Broadcast statistics to all threads
    float mean = shared_sum[0] / out_features;
    float var = shared_sq_sum[0] / out_features - mean * mean;
    float inv_std = rsqrtf(var + eps);
    
    // Apply instance normalization
    float norm_val = (linear_val - mean) * inv_std;
    
    // Apply learnable parameters for normalization
    if (tid < out_features) {
        norm_val = norm_val * norm_weight[tid] + norm_bias[tid];
        
        // Final element-wise operations: (norm_val + y) * y
        float y_val = y[batch_idx * out_features + tid];
        out[batch_idx * out_features + tid] = (norm_val + y_val) * y_val;
    }
}

void fused_op_forward(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor norm_weight,
    torch::Tensor norm_bias,
    torch::Tensor out,
    float eps
) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);
    
    // Launch configuration
    int threads_per_block = min(MAX_THREADS_PER_BLOCK, ((out_features + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE);
    dim3 grid(batch_size);
    dim3 block(threads_per_block);
    
    // Shared memory size (for reduction across warps)
    int num_warps = (threads_per_block + WARP_SIZE - 1) / WARP_SIZE;
    size_t shared_mem_size = 2 * num_warps * sizeof(float);
    
    fused_op_forward_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        norm_weight.data_ptr<float>(),
        norm_bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        eps
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor norm_weight,
    torch::Tensor norm_bias,
    torch::Tensor out,
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
    extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
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
    out = torch.empty_like(y)
    fused_ext.fused_op_forward(
        x, 
        y, 
        bmm_weight, 
        bmm_bias, 
        instance_norm_weight, 
        instance_norm_bias, 
        out, 
        instance_norm_eps
    )
    return out

batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features), torch.rand(batch_size, out_features)]
