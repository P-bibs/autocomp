# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_015726/code_3.py
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

# --- CUDA Kernel (Fused GEMM + Bias + ReLU) ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_DIM 16

__global__ void fused_gemm_bias_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const float* __restrict__ bias,
    int batch_size,
    int in_features,
    int out_features
) {
    // Shared memory for tiling
    __shared__ float shmem_input[TILE_DIM][TILE_DIM];
    __shared__ float shmem_weight[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int tile = 0; tile < (in_features + TILE_DIM - 1) / TILE_DIM; ++tile) {
        // Load input tile
        int input_row = row;
        int input_col = tile * TILE_DIM + tx;
        if (input_row < batch_size && input_col < in_features) {
            shmem_input[ty][tx] = input[input_row * in_features + input_col];
        } else {
            shmem_input[ty][tx] = 0.0f;
        }

        // Load weight tile (transposed access)
        int weight_row = col;
        int weight_col = tile * TILE_DIM + ty;
        if (weight_row < out_features && weight_col < in_features) {
            shmem_weight[ty][tx] = weight[weight_row * in_features + weight_col];
        } else {
            shmem_weight[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += shmem_input[ty][k] * shmem_weight[tx][k];
        }

        __syncthreads();
    }

    // Write result with bias and ReLU
    if (row < batch_size && col < out_features) {
        float val = sum + bias[col];
        output[row * out_features + col] = fmaxf(0.0f, val);
    }
}

void launch_fused_gemm_bias_relu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor bias
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(
        (out_features + TILE_DIM - 1) / TILE_DIM,
        (batch_size + TILE_DIM - 1) / TILE_DIM
    );

    fused_gemm_bias_relu_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
}
"""

# --- C++ Binding ---
cpp_source = r"""
#include <torch/extension.h>

void launch_fused_gemm_bias_relu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor bias
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gemm_bias_relu", &launch_fused_gemm_bias_relu, "Fused GEMM + Bias + ReLU");
}
"""

# Compile the extension
extension = load_inline(
    name='fused_gemm_ext',
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
    # Ensure tensors are on CUDA
    if not x.is_cuda:
        x = x.cuda()
    if not gemm_weight.is_cuda:
        gemm_weight = gemm_weight.cuda()
    if not bias.is_cuda:
        bias = bias.cuda()

    # Create output tensor
    output = torch.empty(x.size(0), gemm_weight.size(0), device=x.device, dtype=x.dtype)

    # Launch fused kernel
    extension.fused_gemm_bias_relu(x, gemm_weight, output, bias)

    return output

# --- Original Helper Functions ---
batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)

def get_init_inputs():
    return [in_features, out_features, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_features)]
