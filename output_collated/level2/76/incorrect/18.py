# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_022315/code_6.py
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
# Fused GEMM + Bias + ReLU CUDA kernel
# ------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BM 8   // number of rows of C that a block computes
#define BN 32  // number of columns of C that a block computes
#define BK 32  // inner tile size (must be a divisor of K)

__global__ void fused_gemm_bias_relu_kernel(
    const float* __restrict__ A,   // M x K  (row-major)
    const float* __restrict__ B,   // N x K  // Transposed: effectively K x N when used as weight
    const float* __restrict__ bias,
    float*       __restrict__ C,   // M x N  (row-major)
    int M, int N, int K)
{
    // Shared memory for the current tiles
    __shared__ float Ashared[BM * BK]; // BM x BK
    __shared__ float Bshared[BK * BN]; // BK x BN

    // Block indices
    const int blockRow = blockIdx.x; // iterates over M
    const int blockCol = blockIdx.y; // iterates over N

    // Thread indices inside the block
    const int tx = threadIdx.x; // column offset (0 … BN-1)
    const int ty = threadIdx.y; // row offset    (0 … BM-1)

    // Global row & column of the output element this thread computes
    const int row = blockRow * BM + ty;
    const int col = blockCol * BN + tx;

    float sum = 0.0f;

    // ------------------------------------------------------------
    // Loop over the K dimension in tiles of size BK
    // ------------------------------------------------------------
    for (int k = 0; k < K; k += BK) {
        // ----------------------------------------------------
        // 1) Load tile of A (BM x BK).  Exactly one element per thread.
        // ----------------------------------------------------
        const int tid = ty * blockDim.x + tx;          // 0 … (BM*BN-1)
        const int aIdx = tid;                          // we have BM*BK == BM*BN == 256 threads
        const int aRow = aIdx / BK;
        const int aCol = aIdx % BK;
        const int ga = blockRow * BM + aRow;
        const int ka = k + aCol;
        if (ga < M && ka < K) {
            Ashared[aRow * BK + aCol] = A[ga * K + ka];
        } else {
            Ashared[aRow * BK + aCol] = 0.0f;
        }

        // ----------------------------------------------------
        // 2) Load tile of B (BK x BN). Each thread loads several elements.
        // ----------------------------------------------------
        const int bTileSize = BK * BN;                // 32*32 = 1024
        const int numThreads = blockDim.x * blockDim.y; // 256
        for (int i = tid; i < bTileSize; i += numThreads) {
            const int bRow = i / BN;
            const int bCol = i % BN;
            const int gb = k + bRow;
            const int cb = blockCol * BN + bCol;
            if (gb < K && cb < N) {
                Bshared[bRow * BN + bCol] = B[gb * N + cb]; // B is (K x N) as read
            } else {
                Bshared[bRow * BN + bCol] = 0.0f;
            }
        }

        __syncthreads();

        // ----------------------------------------------------
        // 3) Compute partial dot product for this tile
        // ----------------------------------------------------
        // Each thread multiplies its row of Ashared with its column of Bshared
        for (int i = 0; i < BK; ++i) {
            sum += Ashared[ty * BK + i] * Bshared[i * BN + tx];
        }

        __syncthreads();
    }

    // ------------------------------------------------------------
    // 4) Add bias and apply ReLU, then store the final result
    // ------------------------------------------------------------
    if (row < M && col < N) {
        float val = sum + bias[col];
        val = fmaxf(val, 0.0f);          // ReLU
        C[row * N + col] = val;
    }
}

// Host wrapper that launches the kernel with the proper grid size
void fused_gemm_bias_relu(
    torch::Tensor A,   // M x K
    torch::Tensor B,   // N x K
    torch::Tensor bias,// N
    torch::Tensor C)   // M x N
{
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);

    dim3 block(BN, BM); // 32 x 8 = 256 threads
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

    fused_gemm_bias_relu_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        bias.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K);
}
"""

# ------------------------------------------------------------
# C++ bindings (PyBind11)
# ------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_gemm_bias_relu(torch::Tensor A, torch::Tensor B,
                          torch::Tensor bias, torch::Tensor C);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gemm_bias_relu", &fused_gemm_bias_relu,
          "Fused GEMM + Bias + ReLU");
}
"""

# ------------------------------------------------------------
# Compile the custom extension
# ------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ------------------------------------------------------------
# User-facing model – matches the original signature
# ------------------------------------------------------------
def functional_model(x, *, gemm_weight, bias):
    """
    Computes  x @ gemm_weight^T + bias  followed by ReLU.
    All arguments are torch tensors on CUDA.
    
    Args:
        x: Input tensor of shape (M, K)
        gemm_weight: Weight matrix of shape (N, K) (transposed during GEMM)
        bias: Bias vector of shape (N,)
        
    Returns:
        Output tensor of shape (M, N) after linear transformation and ReLU
    """
    batch_size = x.size(0)          # M
    out_features = bias.size(0)     # N
    # Allocate output buffer (row-major)
    output = torch.empty((batch_size, out_features),
                         device=x.device, dtype=x.dtype)

    # Launch the fused kernel – this replaces both the cuBLAS call
    # and the separate bias+ReLU kernel.
    fused_ext.fused_gemm_bias_relu(x, gemm_weight, bias, output)
    return output
