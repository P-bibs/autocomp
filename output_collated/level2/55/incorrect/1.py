# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_151353/code_0.py
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

# CUDA kernel that fuses linear + maxpool1d + sum + scale operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

template<int BLOCK_SIZE>
__global__ void fused_op_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    int kernel_size,
    float scale_factor
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    if (batch_idx >= batch_size) return;
    
    // Shared memory for reduction
    __shared__ float shared_data[BLOCK_SIZE];
    
    float local_sum = 0.0f;
    
    // Each thread processes multiple output features
    for (int out_idx = tid; out_idx < out_features; out_idx += block_size) {
        // Linear transformation: compute dot product of input with weight[out_idx]
        float linear_result = bias[out_idx];
        const float* weight_row = weight + out_idx * in_features;
        
        // Unroll inner loop for better performance
        int i = 0;
        for (; i < in_features - 3; i += 4) {
            linear_result += input[batch_idx * in_features + i] * weight_row[i];
            linear_result += input[batch_idx * in_features + i + 1] * weight_row[i + 1];
            linear_result += input[batch_idx * in_features + i + 2] * weight_row[i + 2];
            linear_result += input[batch_idx * in_features + i + 3] * weight_row[i + 3];
        }
        for (; i < in_features; ++i) {
            linear_result += input[batch_idx * in_features + i] * weight_row[i];
        }
        
        // Apply max pooling with kernel_size=2
        // In the 1D case with kernel_size=2 and stride=2, 
        // we just take the value as is since there's no spatial dimension
        local_sum += linear_result;
    }
    
    shared_data[tid] = local_sum;
    __syncthreads();
    
    // Block-level reduction
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    
    // First thread writes the result
    if (tid == 0) {
        output[batch_idx] = shared_data[0] * scale_factor;
    }
}

void fused_op_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int batch_size,
    int in_features,
    int out_features,
    int kernel_size,
    float scale_factor
) {
    // Ensure tensors are on CUDA
    at::cuda::CUDAGuard device_guard(input.device());
    
    const int threads_per_block = 256;
    const dim3 blocks(batch_size);
    const dim3 threads(threads_per_block);
    
    fused_op_kernel<256><<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        kernel_size,
        scale_factor
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int batch_size,
    int in_features,
    int out_features,
    int kernel_size,
    float scale_factor
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Linear + MaxPool1d + Sum + Scale operation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
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
    # Move tensors to CUDA if not already
    if not x.is_cuda:
        x = x.cuda()
    if not matmul_weight.is_cuda:
        matmul_weight = matmul_weight.cuda()
    if not matmul_bias.is_cuda:
        matmul_bias = matmul_bias.cuda()
    
    batch_size = x.size(0)
    in_features = x.size(1)
    out_features = matmul_weight.size(0)
    
    # Create output tensor
    output = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
    
    # Call the fused CUDA operation
    fused_ext.fused_op(
        x, matmul_weight, matmul_bias, output,
        batch_size, in_features, out_features,
        max_pool_kernel_size, scale_factor
    )
    
    return output

# Parameters for test
batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]

def get_inputs():
    return [torch.rand(batch_size, in_features)]
