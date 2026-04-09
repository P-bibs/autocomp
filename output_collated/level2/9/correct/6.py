# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_071635/code_1.py
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
from torch.utils.cpp_extension import load_inline

# CUDA Kernel: Tiled GEMM + Element-wise operations (Fused)
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void fused_op_kernel(const float* __restrict__ x, const float* __restrict__ weight, 
                                const float* __restrict__ bias, float* __restrict__ out,
                                float sub_val, float mul_val, int M, int N, int K) {
    extern __shared__ float tile[];
    float* tile_x = tile;
    float* tile_w = &tile[TILE_SIZE * TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float val = 0.0f;

    for (int k_tile = 0; k_tile < K; k_tile += TILE_SIZE) {
        // Load tiles into shared memory with bounds checking
        int x_idx = row * K + (k_tile + threadIdx.x);
        int w_idx = (k_tile + threadIdx.y) * N + col;
        
        tile_x[threadIdx.y * TILE_SIZE + threadIdx.x] = (row < M && (k_tile + threadIdx.x) < K) ? x[x_idx] : 0.0f;
        tile_w[threadIdx.y * TILE_SIZE + threadIdx.x] = ((k_tile + threadIdx.y) < K && col < N) ? weight[w_idx] : 0.0f;
        
        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; ++k) {
            val += tile_x[threadIdx.y * TILE_SIZE + k] * tile_w[k * TILE_SIZE + threadIdx.x];
        }
        __syncthreads();
    }

    // Apply bias, subtract, multiply, and ReLU
    if (row < M && col < N) {
        float res = (val + bias[col] - sub_val) * mul_val;
        out[row * N + col] = res > 0.0f ? res : 0.0f;
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, 
                      torch::Tensor out, float sub, float mul) {
    int M = x.size(0); 
    int K = x.size(1); 
    int N = weight.size(1);
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    size_t shared_mem = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    
    fused_op_kernel<<<blocks, threads, shared_mem>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
        out.data_ptr<float>(), sub, mul, M, N, K);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, 
                      torch::Tensor out, float sub, float mul);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused GEMM + Elementwise");
}
"""

fused_ext = load_inline(
    name='fused_op', 
    cpp_sources=cpp_source, 
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], 
    with_cuda=True
)

def functional_model(x, *, linear_weight, linear_bias, subtract_value, multiply_value):
    out = torch.empty((x.size(0), linear_weight.size(0)), device=x.device, dtype=x.dtype)
    # The kernel expects weight as [K, N], so we transpose the linear_weight [N, K]
    fused_ext.fused_op(x, linear_weight.t().contiguous(), linear_bias, out, subtract_value, multiply_value)
    return out

batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]

def get_inputs():
    return [torch.rand(batch_size, in_features)]
