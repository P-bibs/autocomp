# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_065701/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'subtract_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['linear_weight', 'linear_bias', 'subtract_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['linear_weight', 'linear_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a matrix multiplication, subtraction, multiplication, and ReLU activation.
    """

    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

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
    # State for linear (nn.Linear)
    if 'linear_weight' in flat_state:
        state_kwargs['linear_weight'] = flat_state['linear_weight']
    else:
        state_kwargs['linear_weight'] = getattr(model.linear, 'weight', None)
    if 'linear_bias' in flat_state:
        state_kwargs['linear_bias'] = flat_state['linear_bias']
    else:
        state_kwargs['linear_bias'] = getattr(model.linear, 'bias', None)
    if 'subtract_value' in flat_state:
        state_kwargs['subtract_value'] = flat_state['subtract_value']
    else:
        state_kwargs['subtract_value'] = getattr(model, 'subtract_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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

# CUDA kernel that fuses linear, subtract, multiply, and relu operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void fused_linear_act_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float subtract_value,
    const float multiply_value,
    const int batch_size,
    const int in_features,
    const int out_features
) {
    const int batch_idx = blockIdx.x;
    const int out_dim = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && out_dim < out_features) {
        float sum = 0.0f;
        
        // Perform linear operation: sum(input * weight) + bias
        for (int i = 0; i < in_features; ++i) {
            sum += input[batch_idx * in_features + i] * weight[out_dim * in_features + i];
        }
        sum += bias[out_dim];
        
        // Subtract, multiply, and apply ReLU
        sum = (sum - subtract_value) * multiply_value;
        sum = fmaxf(0.0f, sum); // ReLU
        
        output[batch_idx * out_features + out_dim] = sum;
    }
}

// Optimized version using shared memory for weight reuse
__global__ void fused_linear_act_kernel_optimized(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float subtract_value,
    const float multiply_value,
    const int batch_size,
    const int in_features,
    const int out_features
) {
    extern __shared__ float shared_weights[];
    
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.x;
    const int out_dim = blockIdx.y * blockDim.x + threadIdx.x;
    
    // Load bias value if valid thread
    float bias_val = 0.0f;
    if (out_dim < out_features) {
        bias_val = bias[out_dim];
    }
    
    float result = bias_val;
    
    // Process input in chunks to utilize shared memory
    for (int chunk = 0; chunk < in_features; chunk += blockDim.x) {
        const int feature_idx = chunk + tid;
        
        // Load weight values into shared memory
        if (out_dim < out_features && feature_idx < in_features) {
            shared_weights[tid] = weight[out_dim * in_features + feature_idx];
        } else {
            shared_weights[tid] = 0.0f;
        }
        __syncthreads();
        
        // Compute partial dot product
        if (batch_idx < batch_size && out_dim < out_features) {
            for (int i = 0; i < blockDim.x && (chunk + i) < in_features; i++) {
                result += input[batch_idx * in_features + chunk + i] * shared_weights[i];
            }
        }
        __syncthreads();
    }
    
    // Apply activation function
    if (batch_idx < batch_size && out_dim < out_features) {
        result = (result - subtract_value) * multiply_value;
        result = fmaxf(0.0f, result); // ReLU
        output[batch_idx * out_features + out_dim] = result;
    }
}

void fused_linear_act_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    float subtract_value,
    float multiply_value
) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);
    
    // Choose kernel based on problem size
    if (in_features <= 1024) {
        // Use simple kernel for smaller inputs
        const int threads_per_block = 256;
        const int blocks_per_grid_y = (out_features + threads_per_block - 1) / threads_per_block;
        
        dim3 grid(batch_size, blocks_per_grid_y);
        dim3 block(threads_per_block);
        
        fused_linear_act_kernel<<<grid, block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            output.data_ptr<float>(),
            subtract_value,
            multiply_value,
            batch_size,
            in_features,
            out_features
        );
    } else {
        // Use optimized kernel with shared memory for larger inputs
        const int threads_per_block = 512;
        const int blocks_per_grid_y = (out_features + threads_per_block - 1) / threads_per_block;
        const int shared_mem_size = threads_per_block * sizeof(float);
        
        dim3 grid(batch_size, blocks_per_grid_y);
        dim3 block(threads_per_block);
        
        fused_linear_act_kernel_optimized<<<grid, block, shared_mem_size>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            output.data_ptr<float>(),
            subtract_value,
            multiply_value,
            batch_size,
            in_features,
            out_features
        );
    }
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_linear_act_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    float subtract_value,
    float multiply_value
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_act_forward", &fused_linear_act_forward, "Fused linear + activation forward pass");
}
"""

# Compile the extension with optimization flags
fused_ext = load_inline(
    name='fused_linear_act',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    linear_weight,
    linear_bias,
    subtract_value,
    multiply_value,
):
    # Ensure tensors are on GPU and have correct dtype
    if not x.is_cuda:
        x = x.cuda()
    if not linear_weight.is_cuda:
        linear_weight = linear_weight.cuda()
    if not linear_bias.is_cuda:
        linear_bias = linear_bias.cuda()
        
    if x.dtype != torch.float32:
        x = x.float()
    if linear_weight.dtype != torch.float32:
        linear_weight = linear_weight.float()
    if linear_bias.dtype != torch.float32:
        linear_bias = linear_bias.float()
    
    # Create output tensor
    batch_size = x.size(0)
    out_features = linear_weight.size(0)
    output = torch.empty(batch_size, out_features, device='cuda', dtype=torch.float32)
    
    # Call fused CUDA kernel
    fused_ext.fused_linear_act_forward(
        x, linear_weight, linear_bias, output, subtract_value, multiply_value
    )
    
    return output

batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]

def get_inputs():
    return [torch.rand(batch_size, in_features)]
