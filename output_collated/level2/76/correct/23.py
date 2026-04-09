# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_023135/code_7.py
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

# -------------------------------------------------------------------------
# CUDA source – tiled GEMM + bias + ReLU
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fusion kernel: Y = ReLU(X @ Wᵀ + bias)
__global__ void fused_gemm_bias_relu_kernel(
    const float* __restrict__ X,   // M x K   (row‑major)
    const float* __restrict__ W_T, // K x N   (row‑major, transposed weight)
    const float* __restrict__ bias,// N
    float* __restrict__ Y,         // M x N   (row‑major)
    int M, int K, int N)
{
    // Tile sizes – 16x16 blocks give 256 threads per block
    constexpr int BM = 16;
    constexpr int BK = 16;
    constexpr int BN = 16;

    // Shared memory for tiles
    __shared__ float s_a[BM][BK]; // A tile
    __shared__ float s_b[BK][BN]; // B tile (W_T)

    // Global indices for this thread
    int row = blockIdx.y * BM + threadIdx.y;
    int col = blockIdx.x * BN + threadIdx.x;

    // Accumulator
    float acc = 0.0f;

    // Loop over the K dimension in blocks of BK
    for (int k = 0; k < K; k += BK) {
        // Load tile from X (A)
        int kx = k + threadIdx.x;
        if (row < M && kx < K) {
            s_a[threadIdx.y][threadIdx.x] = X[row * K + kx];
        } else {
            s_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from transposed weight (B)
        int ky = k + threadIdx.y;
        if (ky < K && col < N) {
            s_b[threadIdx.y][threadIdx.x] = W_T[ky * N + col];
        } else {
            s_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Partial dot product for this tile
        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            acc += s_a[threadIdx.y][i] * s_b[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Add bias and apply ReLU
    if (row < M && col < N) {
        acc += bias[col];
        acc = fmaxf(acc, 0.0f);               // ReLU
        Y[row * N + col] = acc;
    }
}

// C++ wrapper that launches the kernel
void fused_op(torch::Tensor X, torch::Tensor W_T, torch::Tensor bias, torch::Tensor Y) {
    int M = X.size(0);
    int K = X.size(1);
    int N = W_T.size(1);

    const int BM = 16, BN = 16;
    dim3 block(BN, BM);                                   // 16×16 = 256 threads
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);      // cover all output elements

    fused_gemm_bias_relu_kernel<<<grid, block>>>(
        X.data_ptr<float>(),
        W_T.data_ptr<float>(),
        bias.data_ptr<float>(),
        Y.data_ptr<float>(),
        M, K, N);
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11) – exposes fused_op to Python
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op(torch::Tensor X, torch::Tensor W_T, torch::Tensor bias, torch::Tensor Y);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused GEMM + bias + ReLU kernel");
}
"""

# -------------------------------------------------------------------------
# Compile the extension with aggressive optimisation flags
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional model – replaces the original three‑step implementation
# -------------------------------------------------------------------------
def functional_model(x, *, gemm_weight, bias):
    """
    Fused linear + bias + ReLU.
    All arguments are PyTorch tensors; the kernel runs entirely on the GPU.
    """
    # Make sure we work on the same device as the input
    device = x.device

    # Transpose the weight to obtain W_T (K × N) and ensure a contiguous layout
    # for coalesced memory accesses in the kernel.
    weight_t = gemm_weight.t().to(device).contiguous()
    bias = bias.to(device)

    batch = x.shape[0]          # M
    in_features = x.shape[1]    # K
    out_features = gemm_weight.shape[0]   # N

    # Allocate output tensor (M × N)
    out = torch.empty((batch, out_features), dtype=x.dtype, device=device)

    # Invoke the fused CUDA kernel
    fused_ext.fused_op(x, weight_t, bias, out)
    return out


# -------------------------------------------------------------------------
# Helper functions required by the evaluation harness
# -------------------------------------------------------------------------
def get_init_inputs():
    return [8192, 8192, (8192,)]


def get_inputs():
    return [torch.rand(1024, 8192)]
