# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_024220/code_6.py
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
# CUDA source – tiled GEMM fused with bias + ReLU
# ------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tile sizes – 64×64 works well on RTX 2080 Ti
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 64;

__global__ void gemm_bias_relu_kernel(
    const float* __restrict__ A,   // (M, K) row-major
    const float* __restrict__ B,   // (N, K) row-major
    const float* __restrict__ bias,
    float* __restrict__ C,         // (M, N) row-major
    int M, int N, int K)
{
    // Shared memory for the current tiles
    __shared__ float sA[BM][BK];
    __shared__ float sB[BN][BK];  // Note: transposed storage for better access

    // Block indices
    const int bx = blockIdx.x;   // column tile index
    const int by = blockIdx.y;   // row tile index

    // Thread indices inside a tile
    const int tx = threadIdx.x % BK;   // column offset within tile
    const int ty = threadIdx.x / BK;   // row offset within tile

    // Early exit if out of bounds
    if (by * BM + ty >= M || bx * BN + tx >= N) return;

    // Global output coordinates for this thread
    const int row = by * BM + ty;
    const int col = bx * BN + tx;

    // Accumulator for the dot product
    float sum = 0.0f;

    // Loop over the K dimension in tiles of size BK
    for (int kk = 0; kk < K; kk += BK) {
        // Load tile of A (M x K) - each thread loads multiple elements cooperatively
        if (row < M && (kk + tx) < K) {
            sA[ty][tx] = A[row * K + (kk + tx)];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // Load tile of B (N x K) - need to transpose the loading pattern
        if (col < N && (kk + ty) < K) {
            sB[tx][ty] = B[col * K + (kk + ty)];  // Transposed storage
        } else {
            sB[tx][ty] = 0.0f;
        }

        __syncthreads();

        // Partial dot product for this tile
        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            sum += sA[ty][i] * sB[tx][i];  // Corrected indexing for transposed B
        }

        __syncthreads();
    }

    // Only write if within bounds
    if (row < M && col < N) {
        // Add bias and apply ReLU
        sum += bias[col];
        sum = fmaxf(0.0f, sum);   // ReLU

        // Store the final result in row-major order
        C[row * N + col] = sum;
    }
}

// Host wrapper that sets up the grid and launches the kernel
void gemm_bias_relu(
    torch::Tensor x,       // (M, K)
    torch::Tensor weight,  // (N, K)
    torch::Tensor bias,    // (N)
    torch::Tensor output)  // (M, N)
{
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = weight.size(0);

    const float* A = x.data_ptr<float>();
    const float* B = weight.data_ptr<float>();
    const float* b = bias.data_ptr<float>();
    float* C = output.data_ptr<float>();

    // 2-D grid: (N/BN) columns × (M/BM) rows
    dim3 blockDim(BM * BK / BK);     // Threads per block (BK * BM / BK = BM, but we need to match tile size)
    dim3 blockDimActual(BK * BM / BN, BN / BK + 1); // Let's recalculate properly
    // Actually let's use a simpler approach:
    dim3 blockDim(BK * 2, BK / 2);   // 128 x 32 = 4096 threads
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    gemm_bias_relu_kernel<<<gridDim, blockDim>>>(A, B, b, C, M, N, K);
}
"""

# Corrected version with better thread indexing
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tile sizes – 64×64 works well on RTX 2080 Ti
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;  // Reduced for better occupancy

__global__ void gemm_bias_relu_kernel(
    const float* __restrict__ A,   // (M, K) row-major
    const float* __restrict__ B,   // (N, K) row-major
    const float* __restrict__ bias,
    float* __restrict__ C,         // (M, N) row-major
    int M, int N, int K)
{
    // Shared memory for the current tiles
    __shared__ float sA[BM][BK];
    __shared__ float sB[BN][BK];

    // Block indices
    const int bx = blockIdx.x;   // column tile index
    const int by = blockIdx.y;   // row tile index
    const int bz = blockIdx.z;   // batch tile index (if needed)

    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Global output coordinates for this thread
    const int row = by * BM + ty;
    const int col = bx * BN + tx;

    // Accumulator for the dot product
    float sum = 0.0f;

    // Loop over the K dimension in tiles of size BK
    for (int kk = 0; kk < K; kk += BK) {
        // Cooperatively load tiles into shared memory
        if (row < M && (kk + tx) < K) {
            sA[ty][tx] = A[row * K + (kk + tx)];
        } else {
            sA[ty][tx] = 0.0f;
        }

        if (col < N && (kk + ty) < K) {
            sB[ty][tx] = B[col * K + (kk + ty)];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial products
        #pragma unroll 8
        for (int i = 0; i < BK; ++i) {
            sum += sA[ty][i] * sB[tx][i];
        }

        __syncthreads();
    }

    // Only write if within bounds
    if (row < M && col < N) {
        // Add bias and apply ReLU
        sum += bias[col];
        sum = fmaxf(0.0f, sum);   // ReLU

        // Store the final result in row-major order
        C[row * N + col] = sum;
    }
}

// Optimized version with better memory access patterns
__global__ void gemm_bias_relu_kernel_opt(
    const float* __restrict__ A,   // (M, K) row-major
    const float* __restrict__ B,   // (N, K) row-major
    const float* __restrict__ bias,
    float* __restrict__ C,         // (M, N) row-major
    int M, int N, int K)
{
    // Shared memory for tiles
    __shared__ float sA[BK][BM];
    __shared__ float sB[BK][BN];

    // Thread and block indices
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Register accumulator
    float sum = 0.0f;

    // Loop over K dimension
    for (int k = 0; k < K; k += BK) {
        // Load A tile (transposed for coalesced access)
        if (tx < BM && ty < BK) {
            sA[ty][tx] = (by * BM + tx < M && k + ty < K) ? 
                         A[(by * BM + tx) * K + k + ty] : 0.0f;
        }
        
        // Load B tile (transposed for coalesced access)
        if (tx < BN && ty < BK) {
            sB[ty][tx] = (bx * BN + tx < N && k + ty < K) ? 
                         B[(bx * BN + tx) * K + k + ty] : 0.0f;
        }

        __syncthreads();

        // Compute partial products
        #pragma unroll
        for (int i = 0; i < BK; i++) {
            sum += sA[i][ty] * sB[i][tx];
        }

        __syncthreads();
    }

    // Write result with bias and ReLU
    if (tx < BN && ty < BM) {
        int out_row = by * BM + ty;
        int out_col = bx * BN + tx;
        
        if (out_row < M && out_col < N) {
            sum += bias[out_col];
            C[out_row * N + out_col] = fmaxf(sum, 0.0f);
        }
    }
}

// Host wrapper that sets up the grid and launches the kernel
void gemm_bias_relu(
    torch::Tensor x,       // (M, K)
    torch::Tensor weight,  // (N, K)
    torch::Tensor bias,    // (N)
    torch::Tensor output)  // (M, N)
{
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = weight.size(0);

    const float* A = x.data_ptr<float>();
    const float* B = weight.data_ptr<float>();
    const float* b = bias.data_ptr<float>();
    float* C = output.data_ptr<float>();

    // Configure grid and block dimensions
    const int BLOCK_SIZE_M = 64;
    const int BLOCK_SIZE_N = 64;
    const int BLOCK_SIZE_K = 32;

    dim3 blockDim(BLOCK_SIZE_N, BLOCK_SIZE_M);
    dim3 gridDim((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, 
                 (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);

    gemm_bias_relu_kernel_opt<<<gridDim, blockDim>>>(A, B, b, C, M, N, K);
}
"""

# ------------------------------------------------------------
# C++ binding – exposes the function to Python
# ------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void gemm_bias_relu(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_bias_relu", &gemm_bias_relu,
          "Fused GEMM + Bias + ReLU (CUDA)");
}
"""

# ------------------------------------------------------------
# Build the inline extension
# ------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ------------------------------------------------------------
# Functional wrapper that will be imported
# ------------------------------------------------------------
def functional_model(x, *, gemm_weight, bias):
    """
    Computes  y = ReLU(x @ gemm_weight^T + bias).
    All inputs are CUDA tensors; output shape is (batch_size, out_features).
    """
    batch_size = x.size(0)
    out_features = bias.size(0)
    output = torch.empty((batch_size, out_features),
                         device=x.device, dtype=x.dtype)

    fused_ext.gemm_bias_relu(x, gemm_weight, bias, output)
    return output


# ------------------------------------------------------------
# Quick sanity-check (not required for the final submission)
# ------------------------------------------------------------
if __name__ == "__main__":
    batch_size = 1024
    in_features = 8192
    out_features = 8192

    x = torch.rand(batch_size, in_features, device='cuda')
    gemm_weight = torch.rand(out_features, in_features, device='cuda')
    bias = torch.rand(out_features, device='cuda')

    # Warm-up
    for _ in range(5):
        _ = functional_model(x, gemm_weight=gemm_weight, bias=bias)
    torch.cuda.synchronize()

    # Timing
    import time
    start = time.time()
    for _ in range(10):
        _ = functional_model(x, gemm_weight=gemm_weight, bias=bias)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"Average forward pass: {elapsed/10:.4f} s")

    # Verify against PyTorch reference
    ref = torch.relu(torch.nn.functional.linear(x, gemm_weight, bias))
    print("Max |difference|:", (functional_model(x, gemm_weight=gemm_weight, bias=bias) - ref).abs().max().item())
