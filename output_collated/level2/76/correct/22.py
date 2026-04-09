# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_023135/code_6.py
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

# ------------------------------------------------------------
#  CUDA kernel – fused GEMM + bias + ReLU
# ------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int BM = 16;   // block tile height
constexpr int BN = 16;   // block tile width
constexpr int BK = 16;   // inner tile size

__global__ void gemm_bias_relu_kernel(
    const float* __restrict__ A,   // x : (M, K)
    const float* __restrict__ B,   // weight : (N, K)  (row-major)
    const float* __restrict__ bias,// (N)
    float* __restrict__ C,         // output : (M, N)
    int M, int N, int K)
{
    // Shared memory for tiles
    __shared__ float sA[BM * BK]; // (16,16)
    __shared__ float sB[BK * BN]; // (16,16)

    // Global row & column for this thread
    int row = blockIdx.y * BM + threadIdx.y;
    int col = blockIdx.x * BN + threadIdx.x;

    float acc = 0.0f;

    // Loop over K in tiles of size BK
    for (int k = 0; k < K; k += BK) {
        // ---- load tile from A (x) ----
        int aIdx = row * K + (k + threadIdx.x);
        if (row < M && (k + threadIdx.x) < K) {
            sA[threadIdx.y * BK + threadIdx.x] = A[aIdx];
        } else {
            sA[threadIdx.y * BK + threadIdx.x] = 0.0f;
        }

        // ---- load tile from B (weight) ----
        int bIdx = col * K + (k + threadIdx.y);
        if (col < N && (k + threadIdx.y) < K) {
            sB[threadIdx.y * BN + threadIdx.x] = B[bIdx];
        } else {
            sB[threadIdx.y * BN + threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // ---- compute partial dot product ----
        for (int i = 0; i < BK; ++i) {
            acc += sA[threadIdx.y * BK + i] * sB[i * BN + threadIdx.x];
        }

        __syncthreads();
    }

    // ---- add bias and apply ReLU ----
    if (row < M && col < N) {
        acc += bias[col];
        acc = fmaxf(acc, 0.0f);  // Use fast math instruction for ReLU
        C[row * N + col] = acc;
    }
}

void fused_gemm_bias_relu(
    torch::Tensor x,       // (M, K)
    torch::Tensor weight,  // (N, K)
    torch::Tensor bias,    // (N)
    torch::Tensor output)  // (M, N)
{
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = weight.size(0);

    dim3 blockDim(BN, BM); // 16 × 16 = 256 threads per block
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    gemm_bias_relu_kernel<<<gridDim, blockDim>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K);
}
"""

# ------------------------------------------------------------
#  C++ binding (pybind11)
# ------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_gemm_bias_relu(torch::Tensor x, torch::Tensor weight,
                          torch::Tensor bias, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gemm_bias_relu", &fused_gemm_bias_relu,
          "Fused GEMM + Bias + ReLU");
}
"""

# ------------------------------------------------------------
#  Build the extension
# ------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ------------------------------------------------------------
#  Functional model that will be imported
# ------------------------------------------------------------
def functional_model(x, *, gemm_weight, bias):
    """
    Computes: output = ReLU(x @ gemm_weight.T + bias)
    All inputs are CUDA tensors.
    """
    batch_size = x.size(0)
    out_features = bias.size(0)
    # allocate output buffer
    output = torch.empty((batch_size, out_features),
                         device=x.device, dtype=x.dtype)
    # launch fused kernel
    fused_ext.fused_gemm_bias_relu(x, gemm_weight, bias, output)
    return output

# ------------------------------------------------------------
#  Helper functions for the evaluation harness
# ------------------------------------------------------------
def get_init_inputs():
    # Return metadata expected by the benchmark
    return [8192, 8192, (8192,)]

def get_inputs():
    batch_size = 1024
    in_features = 8192
    out_features = 8192
    x = torch.rand(batch_size, in_features, device='cuda')
    gemm_weight = torch.rand(out_features, in_features, device='cuda')
    bias = torch.rand(out_features, device='cuda')
    return [x], {"gemm_weight": gemm_weight, "bias": bias}
