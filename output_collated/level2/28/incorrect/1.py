# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_144949/code_0.py
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

# CUDA kernel code
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAMathCompat.h>

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

__global__ void fused_op_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ bmm_weight,
    const float* __restrict__ bmm_bias,
    const float* __restrict__ instance_norm_weight,
    const float* __restrict__ instance_norm_bias,
    const float* __restrict__ instance_norm_running_mean,
    const float* __restrict__ instance_norm_running_var,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    bool use_input_stats,
    float momentum,
    float eps
) {
    int batch_idx = blockIdx.x;
    int out_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (batch_idx >= batch_size) return;
    
    // Shared memory for reduction operations
    extern __shared__ float shared_mem[];
    float* shared_sum = shared_mem;
    float* shared_sum_sq = shared_mem + (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    
    // Step 1: Linear transformation (F.linear)
    float linear_result = 0.0f;
    if (out_idx < out_features) {
        linear_result = bmm_bias[out_idx];
        for (int i = 0; i < in_features; ++i) {
            linear_result += x[batch_idx * in_features + i] * bmm_weight[out_idx * in_features + i];
        }
    }
    
    // Step 2: Instance normalization - compute mean and variance if needed
    float normalized_value = linear_result;
    
    if (use_input_stats) {
        // Compute sum and sum of squares using block reduction
        float sum = (out_idx < out_features) ? linear_result : 0.0f;
        float sum_sq = (out_idx < out_features) ? linear_result * linear_result : 0.0f;
        
        sum = block_reduce_sum(sum, shared_sum);
        sum_sq = block_reduce_sum(sum_sq, shared_sum_sq);
        
        if (threadIdx.x == 0) {
            shared_sum[0] = sum / out_features;
            shared_sum_sq[0] = sum_sq / out_features;
        }
        __syncthreads();
        
        float mean = shared_sum[0];
        float variance = shared_sum_sq[0] - mean * mean;
        
        if (out_idx < out_features) {
            normalized_value = (linear_result - mean) * rsqrtf(variance + eps);
        }
    } else {
        // Use running statistics
        if (out_idx < out_features) {
            float mean = instance_norm_running_mean[out_idx];
            float variance = instance_norm_running_var[out_idx];
            normalized_value = (linear_result - mean) * rsqrtf(variance + eps);
        }
    }
    
    // Apply learnable parameters and remaining operations
    if (out_idx < out_features) {
        // Apply learnable parameters
        normalized_value = normalized_value * instance_norm_weight[out_idx] + instance_norm_bias[out_idx];
        
        // Step 3 & 4: Addition and multiplication
        int y_idx = batch_idx * out_features + out_idx;
        float add_result = normalized_value + y[y_idx];
        float final_result = add_result * y[y_idx];
        
        output[y_idx] = final_result;
    }
}

void fused_op_forward(
    const float* x,
    const float* y,
    const float* bmm_weight,
    const float* bmm_bias,
    const float* instance_norm_weight,
    const float* instance_norm_bias,
    const float* instance_norm_running_mean,
    const float* instance_norm_running_var,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    bool use_input_stats,
    float momentum,
    float eps,
    int grid_x,
    int grid_y,
    int threads_per_block,
    int shared_mem_size
) {
    dim3 grid(grid_x, grid_y);
    dim3 block(threads_per_block);
    
    fused_op_forward_kernel<<<grid, block, shared_mem_size>>>(
        x, y, bmm_weight, bmm_bias,
        instance_norm_weight, instance_norm_bias,
        instance_norm_running_mean, instance_norm_running_var,
        output, batch_size, in_features, out_features,
        use_input_stats, momentum, eps
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const float* x,
    const float* y,
    const float* bmm_weight,
    const float* bmm_bias,
    const float* instance_norm_weight,
    const float* instance_norm_bias,
    const float* instance_norm_running_mean,
    const float* instance_norm_running_var,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    bool use_input_stats,
    float momentum,
    float eps,
    int grid_x,
    int grid_y,
    int threads_per_block,
    int shared_mem_size
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
    # Ensure inputs are on GPU
    if not x.is_cuda:
        x = x.cuda()
        y = y.cuda()
        bmm_weight = bmm_weight.cuda()
        bmm_bias = bmm_bias.cuda()
        instance_norm_running_mean = instance_norm_running_mean.cuda()
        instance_norm_running_var = instance_norm_running_var.cuda()
        instance_norm_weight = instance_norm_weight.cuda()
        instance_norm_bias = instance_norm_bias.cuda()
    
    batch_size = x.size(0)
    in_features = x.size(1)
    out_features = y.size(1)
    
    # Allocate output tensor
    output = torch.empty_like(y)
    
    # Configure kernel launch parameters
    threads_per_block = min(1024, ((out_features + 31) / 32) * 32)  # Round to nearest multiple of 32
    threads_per_block = int(threads_per_block)
    blocks_per_grid_x = batch_size
    blocks_per_grid_y = (out_features + threads_per_block - 1) // threads_per_block
    
    # Shared memory size for reductions
    warps_per_block = (threads_per_block + 31) // 32
    shared_mem_size = 2 * warps_per_block * 4  # 2 arrays of floats for each warp
    
    # Launch fused kernel
    fused_ext.fused_op(
        x.data_ptr(),
        y.data_ptr(),
        bmm_weight.data_ptr(),
        bmm_bias.data_ptr(),
        instance_norm_weight.data_ptr(),
        instance_norm_bias.data_ptr(),
        instance_norm_running_mean.data_ptr(),
        instance_norm_running_var.data_ptr(),
        output.data_ptr(),
        batch_size,
        in_features,
        out_features,
        instance_norm_use_input_stats,
        instance_norm_momentum,
        instance_norm_eps,
        blocks_per_grid_x,
        blocks_per_grid_y,
        threads_per_block,
        int(shared_mem_size)
    )
    
    return output

batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features), torch.rand(batch_size, out_features)]
