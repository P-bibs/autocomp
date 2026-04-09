# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_070814/code_7.py
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

# ----------------------------------------------------------------------
# CUDA Kernel: Tiled GEMM with Fused Bias, Arithmetic, and ReLU
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void fused_gemm_relu_kernel(
    const float* __restrict__ x,   // (N, K)
    const float* __restrict__ w,   // (M, K)
    const float* __restrict__ b,   // (M)
    float* __restrict__ y,         // (N, M)
    const float sub_val,
    const float mul_val,
    const int M, const int N, const int K
) {
    __shared__ float s_x[TILE_DIM][TILE_DIM];
    __shared__ float s_w[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y; // Output feature index
    int col = blockIdx.x * TILE_DIM + threadIdx.x; // Batch index

    float acc = 0.0f;

    for (int k_tile = 0; k_tile < (K + TILE_DIM - 1) / TILE_DIM; ++k_tile) {
        // Load tile of X (batch_idx, k)
        int k_curr = k_tile * TILE_DIM + threadIdx.x;
        if (col < N && k_curr < K)
            s_x[threadIdx.y][threadIdx.x] = x[col * K + k_curr];
        else
            s_x[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of W (out_feat, k)
        k_curr = k_tile * TILE_DIM + threadIdx.y;
        if (row < M && k_curr < K)
            s_w[threadIdx.y][threadIdx.x] = w[row * K + k_curr];
        else
            s_w[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_DIM; ++i) {
            acc += s_x[i][threadIdx.x] * s_w[threadIdx.y][i];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        float val = acc + (b != nullptr ? b[row] : 0.0f);
        val = (val - sub_val) * mul_val;
        y[col * M + row] = (val > 0.0f) ? val : 0.0f;
    }
}

void fused_op_forward(
    const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& b,
    torch::Tensor& y, float sub_val, float mul_val
) {
    int N = x.size(0);
    int K = x.size(1);
    int M = w.size(0);

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    fused_gemm_relu_kernel<<<grid, threads>>>(
        x.data_ptr<float>(), w.data_ptr<float>(),
        b.numel() ? b.data_ptr<float>() : nullptr,
        y.data_ptr<float>(), sub_val, mul_val, M, N, K
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& b, torch::Tensor& y, float sub_val, float mul_val);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused GEMM kernel");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, linear_weight, linear_bias, subtract_value, multiply_value):
    batch_size, in_features = x.shape
    out_features = linear_weight.shape[0]
    
    y = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(
        x.contiguous(), 
        linear_weight.contiguous(), 
        linear_bias.contiguous() if linear_bias is not None else torch.tensor([], device=x.device),
        y, 
        float(subtract_value), 
        float(multiply_value)
    )
    return y
