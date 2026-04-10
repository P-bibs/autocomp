# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_152808/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'kernel_size', 'scale_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['matmul_weight', 'matmul_bias', 'max_pool_kernel_size', 'max_pool_stride', 'max_pool_padding', 'max_pool_dilation', 'max_pool_ceil_mode', 'max_pool_return_indices', 'scale_factor']
REQUIRED_FLAT_STATE_NAMES = ['matmul_weight', 'matmul_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs matrix multiplication, max pooling, sum, and scaling.
    """

    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.max_pool = nn.MaxPool1d(kernel_size)
        self.scale_factor = scale_factor

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
    # State for matmul (nn.Linear)
    if 'matmul_weight' in flat_state:
        state_kwargs['matmul_weight'] = flat_state['matmul_weight']
    else:
        state_kwargs['matmul_weight'] = getattr(model.matmul, 'weight', None)
    if 'matmul_bias' in flat_state:
        state_kwargs['matmul_bias'] = flat_state['matmul_bias']
    else:
        state_kwargs['matmul_bias'] = getattr(model.matmul, 'bias', None)
    # State for max_pool (nn.MaxPool1d)
    state_kwargs['max_pool_kernel_size'] = model.max_pool.kernel_size
    state_kwargs['max_pool_stride'] = model.max_pool.stride
    state_kwargs['max_pool_padding'] = model.max_pool.padding
    state_kwargs['max_pool_dilation'] = model.max_pool.dilation
    state_kwargs['max_pool_ceil_mode'] = model.max_pool.ceil_mode
    state_kwargs['max_pool_return_indices'] = model.max_pool.return_indices
    if 'scale_factor' in flat_state:
        state_kwargs['scale_factor'] = flat_state['scale_factor']
    else:
        state_kwargs['scale_factor'] = getattr(model, 'scale_factor')
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_op_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    int kernel_size,
    float scale_factor
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Shared memory for intermediate results
    extern __shared__ float shared_mem[];
    float* linear_results = shared_mem;
    float* reduction_buffer = shared_mem + out_features;
    
    // Initialize reduction buffer
    reduction_buffer[tid] = 0.0f;
    __syncthreads();
    
    float local_sum = 0.0f;
    
    // Process features in chunks
    for (int chunk_start = 0; chunk_start < out_features; chunk_start += block_size) {
        int feature_idx = chunk_start + tid;
        
        // Compute linear result
        float linear_val = (feature_idx < out_features) ? bias[feature_idx] : -1e38f;
        if (feature_idx < out_features) {
            for (int i = 0; i < in_features; i++) {
                linear_val += x[batch_idx * in_features + i] * weight[feature_idx * in_features + i];
            }
        }
        
        linear_results[tid] = linear_val;
        __syncthreads();
        
        // Perform max pooling within window
        if (feature_idx < out_features) {
            int window_start = (feature_idx / kernel_size) * kernel_size;
            int window_end = min(window_start + kernel_size, out_features);
            int local_window_start = max(window_start, chunk_start);
            int local_window_end = min(window_end, chunk_start + block_size);
            
            float window_max = -1e38f;
            for (int i = local_window_start; i < local_window_end; i++) {
                int local_idx = i - chunk_start;
                if (local_idx >= 0 && local_idx < block_size) {
                    window_max = fmaxf(window_max, linear_results[local_idx]);
                }
            }
            
            // Only one thread per window contributes to the sum
            if (feature_idx == window_start) {
                local_sum += window_max;
            }
        }
        __syncthreads();
    }
    
    // Store local sum in reduction buffer
    reduction_buffer[tid] = local_sum;
    __syncthreads();
    
    // Perform block-level reduction
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            reduction_buffer[tid] += reduction_buffer[tid + s];
        }
        __syncthreads();
    }
    
    // Write final result
    if (tid == 0) {
        output[batch_idx] = reduction_buffer[0] * scale_factor;
    }
}

void fused_op_forward(
    int batch_size,
    int in_features,
    int out_features,
    int kernel_size,
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    float scale_factor
) {
    dim3 grid(batch_size);
    dim3 block(min(1024, ((out_features + 31) / 32) * 32)); // Round up to nearest multiple of 32, max 1024
    int shared_mem_size = (out_features + block.x) * sizeof(float);
    
    fused_op_kernel<<<grid, block, shared_mem_size>>>(
        x, weight, bias, output,
        batch_size, in_features, out_features,
        kernel_size, scale_factor
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    int batch_size,
    int in_features,
    int out_features,
    int kernel_size,
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    float scale_factor
);

torch::Tensor fused_op(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int kernel_size,
    float scale_factor
) {
    auto batch_size = x.size(0);
    auto in_features = x.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::empty({batch_size}, x.options());
    
    fused_op_forward(
        batch_size,
        in_features,
        out_features,
        kernel_size,
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scale_factor
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused linear + maxpool + sum + scale operation");
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
    *,
    matmul_weight,
    matmul_bias,
    max_pool_kernel_size,
    max_pool_stride,
    max_pool_padding,
    max_pool_dilation,
    max_pool_ceil_mode,
    max_pool_return_indices,
    scale_factor,
):
    return fused_ext.fused_op(x, matmul_weight, matmul_bias, max_pool_kernel_size, scale_factor)

batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]

def get_inputs():
    return [torch.rand(batch_size, in_features)]
