# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_160810/code_1.py
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

__global__ void fused_op_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    int kernel_size,
    int stride,
    int padding,
    float scale_factor
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int blockDimx = blockDim.x;
    
    if (batch_idx >= batch_size) return;
    
    // Shared memory for partial sums
    extern __shared__ float shared_mem[];
    float* shared_results = shared_mem;
    
    const float* x_batch = x + batch_idx * in_features;
    
    // Each thread computes one output element
    for (int out_idx = tid; out_idx < out_features; out_idx += blockDimx) {
        float result = bias[out_idx];
        const float* weight_row = weight + out_idx * in_features;
        
        // Vectorized dot product with unrolling
        int j = 0;
        for (; j < in_features - 3; j += 4) {
            result += x_batch[j] * weight_row[j] +
                      x_batch[j+1] * weight_row[j+1] +
                      x_batch[j+2] * weight_row[j+2] +
                      x_batch[j+3] * weight_row[j+3];
        }
        for (; j < in_features; j++) {
            result += x_batch[j] * weight_row[j];
        }
        
        shared_results[out_idx] = result;
    }
    
    __syncthreads();
    
    // Perform max pooling
    for (int out_idx = tid; out_idx < out_features; out_idx += blockDimx) {
        // Determine pooling window
        int pool_start = out_idx * stride - padding;
        int pool_end = pool_start + kernel_size;
        
        // Clamp to valid range
        pool_start = max(0, pool_start);
        pool_end = min(out_features, pool_end);
        
        // Find maximum in the window
        float max_val = -INFINITY;
        bool valid = false;
        for (int i = pool_start; i < pool_end; i++) {
            max_val = fmaxf(max_val, shared_results[i]);
            valid = true;
        }
        
        if (valid) {
            output[batch_idx * out_features + out_idx] = max_val * scale_factor;
        }
    }
}

void fused_op_forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    float scale_factor
) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);
    
    // Default stride to kernel_size if not specified
    if (stride <= 0) stride = kernel_size;
    
    dim3 grid(batch_size);
    dim3 block(min(1024, out_features));
    int shared_mem_size = out_features * sizeof(float);
    
    fused_op_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        kernel_size,
        stride,
        padding,
        scale_factor
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    float scale_factor
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused linear + max pooling + scale operation");
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
    batch_size = x.shape[0]
    out_features = matmul_weight.shape[0]
    
    # Ensure tensors are on CUDA
    if not x.is_cuda:
        x = x.cuda()
    if not matmul_weight.is_cuda:
        matmul_weight = matmul_weight.cuda()
    if not matmul_bias.is_cuda:
        matmul_bias = matmul_bias.cuda()
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_features, device=x.device, dtype=x.dtype)
    
    # Call the fused CUDA operation
    fused_ext.fused_op(
        x, matmul_weight, matmul_bias, output,
        max_pool_kernel_size, max_pool_stride, max_pool_padding, scale_factor
    )
    
    # Sum along the feature dimension to match original behavior
    result = torch.sum(output, dim=1)
    
    return result

# Parameters for testing
batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]

def get_inputs():
    return [torch.rand(batch_size, in_features)]
