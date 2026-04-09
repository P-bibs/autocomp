# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_020620/code_5.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'bias_shape']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['gemm_weight', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['gemm_weight', 'bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, adds a bias term, and applies ReLU.
    """

    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

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
    # State for gemm (nn.Linear)
    if 'gemm_weight' in flat_state:
        state_kwargs['gemm_weight'] = flat_state['gemm_weight']
    else:
        state_kwargs['gemm_weight'] = getattr(model.gemm, 'weight', None)
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
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

# CUDA kernel for fused Linear + Bias + ReLU operation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define BLOCK_SIZE 16

__global__ void fused_linear_bias_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_features,
    const int out_features
) {
    // Tile dimensions
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;
    
    // Shared memory for input and weight tiles
    __shared__ float sh_input[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sh_weight[BLOCK_SIZE][BLOCK_SIZE];
    
    // Output position
    const int row = by * BLOCK_SIZE + ty;
    const int col = bx * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (in_features + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Load input tile
        const int input_col = t * BLOCK_SIZE + tx;
        if (row < batch_size && input_col < in_features) {
            sh_input[ty][tx] = input[row * in_features + input_col];
        } else {
            sh_input[ty][tx] = 0.0f;
        }
        
        // Load weight tile
        const int weight_row = t * BLOCK_SIZE + ty;
        if (weight_row < in_features && col < out_features) {
            sh_weight[ty][tx] = weight[weight_row * out_features + col];
        } else {
            sh_weight[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += sh_input[ty][k] * sh_weight[k][tx];
        }
        
        __syncthreads();
    }
    
    // Apply bias and ReLU
    if (row < batch_size && col < out_features) {
        float result = sum + bias[col];
        result = fmaxf(result, 0.0f); // ReLU
        output[row * out_features + col] = result;
    }
}

void fused_linear_bias_relu_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int batch_size,
    const int in_features,
    const int out_features
) {
    // Define block and grid dimensions
    const dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid_dim(
        (out_features + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE
    );
    
    // Launch kernel
    fused_linear_bias_relu_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
}
"""

# C++ interface for the CUDA kernel
cpp_source = r"""
#include <torch/extension.h>

void fused_linear_bias_relu_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int batch_size,
    const int in_features,
    const int out_features
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_bias_relu", &fused_linear_bias_relu_forward, "Fused Linear + Bias + ReLU forward pass");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_linear_bias_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    gemm_weight,
    bias,
):
    batch_size = x.size(0)
    in_features = x.size(1)
    out_features = gemm_weight.size(0)
    
    # Create output tensor
    output = torch.empty(batch_size, out_features, device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_linear_bias_relu(
        x.contiguous(),
        gemm_weight.contiguous(),
        bias.contiguous(),
        output,
        batch_size,
        in_features,
        out_features
    )
    
    return output

batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)

def get_init_inputs():
    return [in_features, out_features, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_features)]
