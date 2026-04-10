# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150602/code_3.py
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

# ----------------------------------------------------------------------
# CUDA source – contains the fused kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

// ---------------------------------------------------------------------------
// Fused kernel: GEMM + InstanceNorm + element-wise operations
// Each thread block handles one output feature
// Each thread computes one output element
// ---------------------------------------------------------------------------
__global__ void fused_op_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ norm_weight,
    const float* __restrict__ norm_bias,
    float* __restrict__ out,
    const int batch,
    const int in_features,
    const int out_features,
    const bool use_input_stats,
    const float eps)
{
    const int f = blockIdx.x;  // output feature/channel index
    if (f >= out_features) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    // Shared memory for reduction
    __shared__ float sdata[256];  // Assuming max 256 threads per block
    
    // Phase 1: GEMM for this feature + statistics computation
    float mean_sum = 0.0f;
    float var_sum = 0.0f;
    
    for (int b = tid; b < batch; b += block_size) {
        // Compute linear output for this batch element and feature
        float sum = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < in_features; ++k) {
            sum += x[b * in_features + k] * weight[f * in_features + k];
        }
        float linear_out = sum + bias[f];
        
        // Accumulate for mean calculation
        mean_sum += linear_out;
        
        // Store intermediate result for variance and norm computation
        // We'll recompute linear_out when needed to save shared memory
        sdata[threadIdx.x] = linear_out;
        __syncthreads();
    }
    
    // Reduction for mean
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float mean = 0.0f;
    if (use_input_stats && tid == 0) {
        mean = sdata[0] / batch;
        // Store mean in shared memory for variance calculation
        sdata[0] = mean;
    }
    __syncthreads();
    
    if (use_input_stats) {
        mean = sdata[0];  // All threads read the computed mean
        
        // Compute variance
        for (int b = tid; b < batch; b += block_size) {
            float linear_out = 0.0f;
            #pragma unroll 8
            for (int k = 0; k < in_features; ++k) {
                linear_out += x[b * in_features + k] * weight[f * in_features + k];
            }
            linear_out += bias[f];
            
            float diff = linear_out - mean;
            var_sum += diff * diff;
        }
        
        // Reduction for variance
        sdata[tid] = var_sum;
        __syncthreads();
        
        for (int s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
    }
    
    float var = 0.0f;
    if (use_input_stats && tid == 0) {
        var = sdata[0] / batch;
        // Store variance in shared memory
        sdata[1] = var;
    }
    __syncthreads();
    
    if (use_input_stats) {
        var = sdata[1];  // All threads read the computed variance
    } else {
        mean = running_mean[f];
        var = running_var[f];
    }
    
    // Phase 2: Apply normalization and element-wise operations
    float inv_std = rsqrtf(var + eps);
    
    for (int b = tid; b < batch; b += block_size) {
        // Recompute linear output
        float sum = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < in_features; ++k) {
            sum += x[b * in_features + k] * weight[f * in_features + k];
        }
        float linear_out = sum + bias[f];
        
        // Apply InstanceNorm
        float norm_out = (linear_out - mean) * inv_std * norm_weight[f] + norm_bias[f];
        
        // Apply element-wise operations: (norm + y) * y
        float y_val = y[b * out_features + f];
        out[b * out_features + f] = (norm_out + y_val) * y_val;
    }
}

// ---------------------------------------------------------------------------
// Optimized version using warp-level primitives
// ---------------------------------------------------------------------------
__global__ void fused_op_kernel_opt(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ norm_weight,
    const float* __restrict__ norm_bias,
    float* __restrict__ out,
    const int batch,
    const int in_features,
    const int out_features,
    const bool use_input_stats,
    const float eps)
{
    const int f = blockIdx.x;  // output feature/channel index
    if (f >= out_features) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Shared memory for reduction
    __shared__ float s_mean, s_var;
    
    // Phase 1: GEMM for this feature + statistics computation
    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;
    
    // Each thread processes multiple batch elements
    for (int b = tid; b < batch; b += blockDim.x) {
        // Compute linear output for this batch element and feature
        float sum = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < in_features; ++k) {
            sum += x[b * in_features + k] * weight[f * in_features + k];
        }
        float linear_out = sum + bias[f];
        
        // Accumulate for mean calculation
        thread_sum += linear_out;
        thread_sum_sq += linear_out * linear_out;
    }
    
    // Warp-level reductions
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);
        thread_sum_sq += __shfl_down_sync(0xFFFFFFFF, thread_sum_sq, offset);
    }
    
    // Store warp results in shared memory
    __shared__ float warp_sums[32];
    __shared__ float warp_sum_sqs[32];
    
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
        warp_sum_sqs[warp_id] = thread_sum_sq;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (lane_id == 0 && use_input_stats) {
            s_mean = val / batch;
        }
        
        val = (lane_id < (blockDim.x + 31) / 32) ? warp_sum_sqs[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (lane_id == 0 && use_input_stats) {
            float mean_sq = val / batch;
            s_var = mean_sq - s_mean * s_mean;
        }
    }
    __syncthreads();
    
    // Get mean and variance
    float mean, var;
    if (use_input_stats) {
        mean = s_mean;
        var = s_var;
    } else {
        mean = running_mean[f];
        var = running_var[f];
    }
    
    // Phase 2: Apply normalization and element-wise operations
    float inv_std = rsqrtf(var + eps);
    
    for (int b = tid; b < batch; b += blockDim.x) {
        // Recompute linear output
        float sum = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < in_features; ++k) {
            sum += x[b * in_features + k] * weight[f * in_features + k];
        }
        float linear_out = sum + bias[f];
        
        // Apply InstanceNorm
        float norm_out = (linear_out - mean) * inv_std * norm_weight[f] + norm_bias[f];
        
        // Apply element-wise operations: (norm + y) * y
        float y_val = y[b * out_features + f];
        out[b * out_features + f] = (norm_out + y_val) * y_val;
    }
}

// ---------------------------------------------------------------------------
// Host wrapper
// ---------------------------------------------------------------------------
void fused_op_forward(
    const torch::Tensor& x,
    const torch::Tensor& y,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& running_mean,
    const torch::Tensor& running_var,
    const torch::Tensor& norm_weight,
    const torch::Tensor& norm_bias,
    torch::Tensor& out,
    const bool use_input_stats,
    const float eps)
{
    const int batch = x.size(0);
    const int in_features = x.size(1);
    const int out_features = weight.size(0);
    
    // Launch configuration
    const int threads_per_block = 256;
    const int blocks = out_features;
    
    fused_op_kernel_opt<<<blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        norm_weight.data_ptr<float>(),
        norm_bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        in_features,
        out_features,
        use_input_stats,
        eps
    );
}
"""

# ----------------------------------------------------------------------
# C++ source – pybind11 bindings
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const torch::Tensor& x,
    const torch::Tensor& y,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& running_mean,
    const torch::Tensor& running_var,
    const torch::Tensor& norm_weight,
    const torch::Tensor& norm_bias,
    torch::Tensor& out,
    const bool use_input_stats,
    const float eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op_forward", &fused_op_forward, "Fused operation forward pass");
}
"""

# ----------------------------------------------------------------------
# Build the inline CUDA extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Original helper functions (kept for completeness)
# ----------------------------------------------------------------------
batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features),
            torch.rand(batch_size, out_features)]

# ----------------------------------------------------------------------
# The new functional_model – replaces all PyTorch ops with custom CUDA kernels
# ----------------------------------------------------------------------
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
    # Ensure all inputs are on the GPU
    if not x.is_cuda:
        x = x.cuda()
    if not y.is_cuda:
        y = y.cuda()

    weight = bmm_weight.cuda().contiguous()
    bias   = bmm_bias.cuda().contiguous()
    running_mean = instance_norm_running_mean.cuda().contiguous()
    running_var  = instance_norm_running_var.cuda().contiguous()
    norm_weight  = instance_norm_weight.cuda().contiguous()
    norm_bias    = instance_norm_bias.cuda().contiguous()

    batch   = x.size(0)
    N       = out_features

    # Allocate output tensor
    out = torch.empty((batch, N), dtype=x.dtype, device='cuda')
    
    # Call fused operation
    fused_ext.fused_op_forward(
        x, y,
        weight, bias,
        running_mean, running_var,
        norm_weight, norm_bias,
        out,
        instance_norm_use_input_stats,
        instance_norm_eps
    )
    
    return out
