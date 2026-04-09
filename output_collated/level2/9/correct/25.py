# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_075823/code_7.py
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

# -------------------------------------------------------------------------
# CUDA kernel: Fused GEMM (Tiled) + Bias + Post-Processing (Subtract/Multiply/ReLU)
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tile sizes optimized for occupancy and shared memory on RTX 2080Ti
constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 32;

__global__ void fused_op_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const float sub_val,
    const float mul_val,
    const int M, const int K, const int N) 
{
    // Shared memory for tiles
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    int row = blockIdx.y * BM + threadIdx.y;
    int col = blockIdx.x * BN + threadIdx.x;

    float sum = 0.0f;

    // Tiled Matrix Multiplication
    for (int k = 0; k < K; k += BK) {
        // Load x tile (coalesced)
        if (row < M && (k + threadIdx.x) < K)
            sA[threadIdx.y][threadIdx.x] = x[row * K + (k + threadIdx.x)];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load w tile (transposed weight)
        // w is [N, K], we access col [N] and k [K]
        if (col < N && (k + threadIdx.y) < K)
            sB[threadIdx.y][threadIdx.x] = w[col * K + (k + threadIdx.y)];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }

    // Apply bias and post-processing
    if (row < M && col < N) {
        float val = sum + bias[col];
        val = (val - sub_val) * mul_val;
        out[row * N + col] = (val > 0.0f) ? val : 0.0f;
    }
}

void fused_op_forward(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor bias,
    torch::Tensor out,
    float sub_val,
    float mul_val)
{
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = w.size(0);

    dim3 threads(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    fused_op_kernel<<<grid, threads>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        sub_val, mul_val, M, K, N);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor x, torch::Tensor w, torch::Tensor bias, torch::Tensor out, float sub_val, float mul_val);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused linear + post-proc kernel");
}
"""

# Compile
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

def functional_model(
    x,
    *,
    linear_weight,
    linear_bias,
    subtract_value,
    multiply_value,
):
    # Ensure inputs are contiguous on GPU
    x = x.contiguous().cuda()
    w = linear_weight.contiguous().cuda()
    b = linear_bias.contiguous().cuda()
    
    out = torch.empty((x.size(0), w.size(0)), device='cuda', dtype=torch.float32)
    
    fused_ext.fused_op(
        x, w, b, out, float(subtract_value), float(multiply_value)
    )
    return out

# Global Config
batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]

def get_inputs():
    return [torch.rand(batch_size, in_features)]
