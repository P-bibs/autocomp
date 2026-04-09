# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_024220/code_7.py
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

# ----------------------------------------------------------------------
# Inline CUDA code – fused GEMM + bias + ReLU kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ------------------------------------------------------------------
// Tiled GEMM with fused bias addition and ReLU
// ------------------------------------------------------------------
__global__ void fused_gemm_relu_kernel(
    const float* __restrict__ A,   // M x K   (row-major)
    const float* __restrict__ B,   // K x N   (row-major, i.e. weight^T)
    const float* __restrict__ bias,// N
    float*       __restrict__ C,   // M x N   (output)
    int M, int N, int K)
{
    // Tile sizes – same as block dimensions
    const int BM = 16;
    const int BN = 16;
    const int BK = 16;

    // Shared memory for the two tiles
    __shared__ float As[BM * BK]; // A-tile
    __shared__ float Bs[BK * BN]; // B-tile

    // Thread indices inside the block
    const int tx = threadIdx.x; // 0 … BN-1
    const int ty = threadIdx.y; // 0 … BM-1

    // Output coordinates
    const int row = blockIdx.y * BM + ty; // global row in C
    const int col = blockIdx.x * BN + tx; // global column in C

    float acc = 0.0f;

    // Only threads that belong to a valid output element do work
    if (row < M && col < N) {
        // Loop over the K dimension in blocks of BK
        for (int k = 0; k < K; k += BK) {
            // ---- load A-tile (row-major) ----
            int aCol = k + tx;
            if (row < M && aCol < K) {
                // Coalesced read from A
                As[ty * BK + tx] = A[row * K + aCol];
            } else {
                As[ty * BK + tx] = 0.0f;
            }

            // ---- load B-tile (K-major) ----
            int bRow = k + ty;
            if (bRow < K && col < N) {
                // Coalesced read from B (weight^T)
                Bs[ty * BN + tx] = B[bRow * N + col];
            } else {
                Bs[ty * BN + tx] = 0.0f;
            }

            __syncthreads();

            // ---- inner product for this tile ----
            for (int kk = 0; kk < BK; ++kk) {
                acc += As[ty * BK + kk] * Bs[kk * BN + tx];
            }

            __syncthreads();
        }

        // ---- bias addition + ReLU ----
        float val = acc + bias[col];
        if (val < 0.0f) val = 0.0f;
        C[row * N + col] = val;
    }
}

// ------------------------------------------------------------------
// Host wrapper called from Python
// ------------------------------------------------------------------
void fused_gemm_relu(torch::Tensor A,
                     torch::Tensor B,
                     torch::Tensor bias,
                     torch::Tensor C)
{
    // Ensure tensors are contiguous and have the expected layout
    A = A.contiguous();
    B = B.contiguous();
    bias = bias.contiguous();
    C = C.contiguous();

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    const int BM = 16;
    const int BN = 16;
    dim3 block(BN, BM);                // block.x = 16, block.y = 16
    dim3 grid((N + BN - 1) / BN,
              (M + BM - 1) / BM);      // enough blocks to cover the output

    fused_gemm_relu_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        bias.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K);
}
"""

# ----------------------------------------------------------------------
# C++ binding – creates the Python entry point
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_gemm_relu(torch::Tensor A,
                     torch::Tensor B,
                     torch::Tensor bias,
                     torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gemm_relu", &fused_gemm_relu,
          "Fused GEMM + bias addition + ReLU kernel");
}
"""

# ----------------------------------------------------------------------
# Compile the extension (CUDA + C++) with aggressive optimisation flags
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# ----------------------------------------------------------------------
# The functional model that will be evaluated
# ----------------------------------------------------------------------
def functional_model(
    x,               # tensor of shape (batch, in_features)
    *,
    gemm_weight,    # tensor of shape (out_features, in_features)
    bias,           # tensor of shape (out_features,)
):
    # Ensure we work in FP32 (the original model uses FP32)
    x = x.float()
    bias = bias.float()

    # Transpose the weight to K×N (K=in_features, N=out_features)
    # This gives coalesced memory accesses in the kernel.
    weight_T = gemm_weight.t().contiguous()

    # Allocate output tensor
    batch = x.size(0)
    out_features = gemm_weight.size(0)
    out = torch.empty((batch, out_features), dtype=x.dtype, device=x.device)

    # Call the fused CUDA kernel
    fused_ext.fused_gemm_relu(x, weight_T, bias, out)

    return out


# ----------------------------------------------------------------------
# Helpers required by the evaluation harness (they are not part of the
# performance-critical path)
# ----------------------------------------------------------------------
def get_init_inputs():
    # Example dimensions used in the original benchmark
    return [8192, 8192, (8192,)]


def get_inputs():
    # Same batch size and feature size as in the original code
    return [torch.rand(1024, 8192, device='cuda')]
