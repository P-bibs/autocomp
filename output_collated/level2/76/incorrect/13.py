# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_021357/code_5.py
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

# --- CUDA Kernel Code ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_SIZE 32

__global__ void fused_linear_bias_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_features,
    const int out_features
) {
    // Tile-based GEMM with bias addition and ReLU
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * blockDim.y + ty;
    const int col = blockIdx.x * blockDim.x + tx;

    // Shared memory for input and weight tiles
    __shared__ float sh_input[TILE_SIZE][TILE_SIZE];
    __shared__ float sh_weight[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (in_features + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load input tile
        if (row < batch_size && t * TILE_SIZE + tx < in_features) {
            sh_input[ty][tx] = input[row * in_features + t * TILE_SIZE + tx];
        } else {
            sh_input[ty][tx] = 0.0f;
        }

        // Load weight tile (transposed)
        if (col < out_features && t * TILE_SIZE + ty < in_features) {
            sh_weight[tx][ty] = weight[(t * TILE_SIZE + ty) * out_features + col];
        } else {
            sh_weight[tx][ty] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sh_input[ty][k] * sh_weight[tx][k];
        }

        __syncthreads();
    }

    // Write result with bias addition and ReLU
    if (row < batch_size && col < out_features) {
        float val = sum + bias[col];
        output[row * out_features + col] = fmaxf(val, 0.0f);
    }
}

void fused_linear_bias_relu_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output
) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    // Define block and grid dimensions
    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid(
        (out_features + block.x - 1) / block.x,
        (batch_size + block.y - 1) / block.y
    );

    fused_linear_bias_relu_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
}
"""

# --- C++ Interface/Bindings ---
cpp_source = r"""
#include <torch/extension.h>

void fused_linear_bias_relu_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_bias_relu_forward", &fused_linear_bias_relu_forward, "Fused Linear+Bias+ReLU forward");
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
    *,
    gemm_weight,
    bias,
):
    # Create output tensor
    output = torch.empty(x.size(0), gemm_weight.size(0), dtype=x.dtype, device=x.device)
    
    # Call fused CUDA kernel
    fused_ext.fused_linear_bias_relu_forward(x, gemm_weight, bias, output)
    
    return output

batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)

def get_init_inputs():
    return [in_features, out_features, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_features)]
