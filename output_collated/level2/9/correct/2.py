# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_065701/code_3.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Fused CUDA kernel source
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_SIZE 32

__global__ void fused_op_forward_kernel(
    const float* __restrict__ x_ptr,
    const float* __restrict__ weight_ptr,
    const float* __restrict__ bias_ptr,
    float* output_ptr,
    int batch_size,
    int in_features,
    int out_features,
    float subtract_value,
    float multiply_value
) {
    // Shared memory for tile-based computation
    __shared__ float x_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float weight_shared[TILE_SIZE][TILE_SIZE];
    
    int batch_idx = blockIdx.z;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (in_features + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load x tile into shared memory
        int x_row = out_row;
        int x_col = tile * TILE_SIZE + threadIdx.x;
        if (x_row < batch_size && x_col < in_features) {
            x_shared[threadIdx.y][threadIdx.x] = x_ptr[x_row * in_features + x_col];
        } else {
            x_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load weight tile into shared memory
        int w_row = out_col;
        int w_col = tile * TILE_SIZE + threadIdx.y;
        if (w_row < out_features && w_col < in_features) {
            weight_shared[threadIdx.x][threadIdx.y] = weight_ptr[w_row * in_features + w_col];
        } else {
            weight_shared[threadIdx.x][threadIdx.y] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += x_shared[threadIdx.y][k] * weight_shared[threadIdx.x][k];
        }
        
        __syncthreads();
    }
    
    // Apply bias, subtraction, multiplication, and ReLU
    if (out_row < batch_size && out_col < out_features) {
        sum += bias_ptr[out_col];
        sum = (sum - subtract_value) * multiply_value;
        sum = fmaxf(0.0f, sum); // ReLU
        
        output_ptr[out_row * out_features + out_col] = sum;
    }
}

void fused_op_forward(
    const float* x_ptr,
    const float* weight_ptr,
    const float* bias_ptr,
    float* output_ptr,
    int batch_size,
    int in_features,
    int out_features,
    float subtract_value,
    float multiply_value
) {
    // Use 3D grid: (output_features, batch_size, 1)
    dim3 block_size(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid_size(
        (out_features + TILE_SIZE - 1) / TILE_SIZE,
        (batch_size + TILE_SIZE - 1) / TILE_SIZE,
        1
    );
    
    fused_op_forward_kernel<<<grid_size, block_size>>>(
        x_ptr, weight_ptr, bias_ptr, output_ptr,
        batch_size, in_features, out_features,
        subtract_value, multiply_value
    );
}
"""

# C++ bindings
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the function in the .cu file
void fused_op_forward(
    const float* x_ptr,
    const float* weight_ptr,
    const float* bias_ptr,
    float* output_ptr,
    int batch_size,
    int in_features,
    int out_features,
    float subtract_value,
    float multiply_value
);

torch::Tensor fused_op(
    torch::Tensor x,
    torch::Tensor linear_weight,
    torch::Tensor linear_bias,
    float subtract_value,
    float multiply_value
) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = linear_weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, x.options());
    
    fused_op_forward(
        x.data_ptr<float>(),
        linear_weight.data_ptr<float>(),
        linear_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_features, out_features,
        subtract_value, multiply_value
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused linear + subtract + multiply + ReLU operation");
}
"""

# Compile the fused extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]

def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda', dtype=torch.float32)]

def functional_model(
    x,
    *,
    linear_weight,
    linear_bias,
    subtract_value,
    multiply_value,
):
    # Use the fused CUDA kernel instead of separate PyTorch operations
    return fused_ext.fused_op(
        x, linear_weight, linear_bias, subtract_value, multiply_value
    )
