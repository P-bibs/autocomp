# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_155300/code_1.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_op_forward_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    float scale_factor
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    extern __shared__ float shared_mem[];
    float* linear_results = shared_mem;
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Phase 1: Compute linear transformation for this batch
    const float* input_batch = input + batch_idx * in_features;
    
    // Each thread computes multiple output elements
    for (int out_idx = tid; out_idx < out_features; out_idx += block_size) {
        float result = bias[out_idx];
        const float* weight_row = weight + out_idx * in_features;
        
        // Compute dot product
        for (int i = 0; i < in_features; i++) {
            result += weight_row[i] * input_batch[i];
        }
        linear_results[out_idx] = result;
    }
    
    __syncthreads();
    
    // Phase 2: Max pooling + sum reduction
    if (tid == 0) {
        // Calculate output size after pooling
        // Formula: output_size = floor((input_size + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
        int input_size = out_features;
        int output_size = (input_size + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1;
        if (output_size <= 0) output_size = 1;
        
        float sum_result = 0.0f;
        
        // Apply max pooling with given parameters and sum
        for (int out_idx = 0; out_idx < output_size; out_idx++) {
            float max_val = -INFINITY;
            
            // Calculate the start position in the input
            int start = out_idx * stride - padding;
            
            // Check each position in the kernel
            for (int k = 0; k < kernel_size; k++) {
                int idx = start + k * dilation;
                if (idx >= 0 && idx < input_size) {
                    max_val = fmaxf(max_val, linear_results[idx]);
                }
            }
            sum_result += max_val;
        }
        
        output[batch_idx] = sum_result * scale_factor;
    }
}

void fused_op_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    float scale_factor
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    dim3 grid(batch_size);
    dim3 block(min(1024, (out_features + 31) / 32 * 32));  // Round up to multiple of 32, max 1024
    
    size_t shared_mem_size = out_features * sizeof(float);
    
    fused_op_forward_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        kernel_size,
        stride,
        padding,
        dilation,
        scale_factor
    );
    
    cudaDeviceSynchronize();
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the function in the .cu file
void fused_op_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    float scale_factor
);

torch::Tensor fused_op(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    float scale_factor
) {
    auto output = torch::empty({input.size(0)}, input.options());
    fused_op_forward(input, weight, bias, output, kernel_size, stride, padding, dilation, scale_factor);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused linear + max_pool1d + sum + scale operation");
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
    # Handle default values for max pooling parameters
    if max_pool_stride == []:
        max_pool_stride = max_pool_kernel_size
    else:
        max_pool_stride = max_pool_stride[0] if isinstance(max_pool_stride, list) else max_pool_stride
        
    max_pool_padding = max_pool_padding[0] if isinstance(max_pool_padding, list) else max_pool_padding
    max_pool_dilation = max_pool_dilation[0] if isinstance(max_pool_dilation, list) else max_pool_dilation
    
    # Move tensors to CUDA if not already
    if not x.is_cuda:
        x = x.cuda()
    if not matmul_weight.is_cuda:
        matmul_weight = matmul_weight.cuda()
    if not matmul_bias.is_cuda:
        matmul_bias = matmul_bias.cuda()
        
    # Call the fused operation
    return fused_ext.fused_op(
        x, matmul_weight, matmul_bias, 
        max_pool_kernel_size, max_pool_stride, max_pool_padding, max_pool_dilation, scale_factor
    )

batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]

def get_inputs():
    return [torch.rand(batch_size, in_features)]
