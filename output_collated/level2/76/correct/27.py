# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_023135/code_15.py
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
from torch.utils.cpp_extension import load_inline

# CUDA source – Tiled GEMM with Bias and ReLU fusion
# Uses 32x32 tiles for better hardware utilization on Turing architecture
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_gemm_bias_relu_kernel(
    const float* __restrict__ X,   // M x K
    const float* __restrict__ W_T, // K x N
    const float* __restrict__ bias,// N
    float* __restrict__ Y,         // M x N
    int M, int K, int N)
{
    const int TILE_SIZE = 32;
    __shared__ float s_x[TILE_SIZE][TILE_SIZE];
    __shared__ float s_w[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;

    for (int k_step = 0; k_step < (K + TILE_SIZE - 1) / TILE_SIZE; ++k_step) {
        // Load tiles into shared memory
        if (row < M && (k_step * TILE_SIZE + threadIdx.x) < K)
            s_x[threadIdx.y][threadIdx.x] = X[row * K + (k_step * TILE_SIZE + threadIdx.x)];
        else
            s_x[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (k_step * TILE_SIZE + threadIdx.y) < K)
            s_w[threadIdx.y][threadIdx.x] = W_T[(k_step * TILE_SIZE + threadIdx.y) * N + col];
        else
            s_w[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            acc += s_x[threadIdx.y][i] * s_w[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        float val = acc + bias[col];
        Y[row * N + col] = (val > 0.0f) ? val : 0.0f;
    }
}

void fused_op_cuda(torch::Tensor X, torch::Tensor W_T, torch::Tensor bias, torch::Tensor Y) {
    int M = X.size(0);
    int K = X.size(1);
    int N = W_T.size(1);
    
    dim3 threads(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);

    fused_gemm_bias_relu_kernel<<<grid, threads>>>(
        X.data_ptr<float>(), W_T.data_ptr<float>(), bias.data_ptr<float>(), Y.data_ptr<float>(), M, K, N
    );
}
"""

cpp_source = r"""
void fused_op_cuda(torch::Tensor X, torch::Tensor W_T, torch::Tensor bias, torch::Tensor Y);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_cuda, "Fused GEMM + Bias + ReLU");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, gemm_weight, bias):
    device = x.device
    # Ensure weight is Wᵀ (transposed KxN) for coalesced access
    # We cache the transposed weight if this were a production system, 
    # but for this harness, we ensure the correct layout here.
    weight_t = gemm_weight.t().contiguous()
    
    M, K = x.shape
    N = weight_t.shape[1]
    
    out = torch.empty((M, N), dtype=x.dtype, device=device)
    fused_ext.fused_op(x, weight_t, bias, out)
    return out

def get_init_inputs():
    return [8192, 8192, (8192,)]

def get_inputs():
    return [torch.rand(1024, 8192, device='cuda')]
