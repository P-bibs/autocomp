# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_105411/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['linear_weight', 'linear_bias']
REQUIRED_FLAT_STATE_NAMES = ['linear_weight', 'linear_bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Mish, and applies Mish again.
    """

    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

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
# CUDA source – fused linear + double mish kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 32;

// Fast Mish: x * tanh(log1p(exp(x)))
__device__ __forceinline__ float mish(float x) {
    float e = __expf(x);
    float sp = log1pf(e);          // softplus
    float th = tanhf(sp);
    return x * th;
}

// Tiled GEMM with fused bias and double mish
__global__ void fused_op_kernel(
    const float* __restrict__ A,   // input (M x K)
    const float* __restrict__ B,   // weight (N x K)
    const float* __restrict__ bias,
    float* __restrict__ C,         // output (M x N)
    int M, int N, int K)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Output coordinates for this thread
    int row = by * BM + ty;
    int col = bx * BN + tx;

    // Accumulator in register
    float sum = 0.0f;

    // Shared memory tiles
    __shared__ float tileA[BM * BK];
    __shared__ float tileB[BK * BN];

    // Loop over K in blocks of BK
    for (int k = 0; k < K; k += BK) {
        // ---- load tile from A (input) ----
        int aRow = row;
        int aCol = k + tx;
        if (aRow < M && aCol < K) {
            tileA[ty * BK + tx] = A[aRow * K + aCol];
        } else {
            tileA[ty * BK + tx] = 0.0f;
        }

        // ---- load tile from B (weight) ----
        int bRow = k + ty;
        int bCol = col;
        if (bRow < K && bCol < N) {
            tileB[ty * BN + tx] = B[bRow * N + bCol];
        } else {
            tileB[ty * BN + tx] = 0.0f;
        }

        __syncthreads();

        // ---- compute partial dot‑product ----
        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            sum += tileA[ty * BK + i] * tileB[i * BN + tx];
        }

        __syncthreads();
    }

    // ---- add bias, apply Mish twice, store result ----
    if (row < M && col < N) {
        sum += bias[col];
        sum = mish(sum);
        sum = mish(sum);
        C[row * N + col] = sum;
    }
}

// Host wrapper that launches the kernel
void fused_op_forward_kernel_launcher(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output) {
    
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = weight.size(0);

    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    const int shared_mem = (BM * BK + BK * BN) * sizeof(float);

    fused_op_kernel<<<grid, block, shared_mem>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR(cudaGetErrorString(err));
    }
}
"""

# ----------------------------------------------------------------------
# C++ bindings – expose fused_op to Python
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward_kernel_launcher(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output);

void fused_op_forward(const torch::Tensor& x,
                      const torch::Tensor& weight,
                      const torch::Tensor& bias,
                      torch::Tensor& output) {
    fused_op_forward_kernel_launcher(x, weight, bias, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op_forward", &fused_op_forward, "Fused linear + mish*2 forward pass");
}
"""

# ----------------------------------------------------------------------
# Compile the inline CUDA extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional model – the entry point that will be imported
# ----------------------------------------------------------------------
def functional_model(x, *, linear_weight, linear_bias):
    """
    Fused linear layer (matrix multiply + bias) followed by two Mish activations.
    All three operations are performed in a single custom CUDA kernel.
    """
    # Ensure inputs reside on the GPU (the evaluation harness expects GPU tensors)
    if not x.is_cuda:
        x = x.cuda()
    if not linear_weight.is_cuda:
        linear_weight = linear_weight.cuda()
    if not linear_bias.is_cuda:
        linear_bias = linear_bias.cuda()

    # Create output tensor
    M = x.size(0)
    N = linear_weight.size(0)
    output = torch.empty((M, N), dtype=x.dtype, device=x.device)
    
    # Call the fused CUDA extension
    fused_ext.fused_op_forward(x, linear_weight, linear_bias, output)
    return output


# ----------------------------------------------------------------------
# Helper functions required by the harness (not used in the kernel)
# ----------------------------------------------------------------------
def get_init_inputs():
    return [8192, 8192]   # in_features, out_features

def get_inputs():
    return [torch.rand(1024, 8192, device='cuda')]

# ----------------------------------------------------------------------
# Quick sanity check (can be removed when submitting)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple test to verify correctness and shape
    batch = 1024
    inc   = 8192
    outc  = 8192
    x     = torch.rand(batch, inc, device='cuda')
    w     = torch.rand(outc, inc, device='cuda')
    b     = torch.rand(outc, device='cuda')

    y = functional_model(x, linear_weight=w, linear_bias=b)
    print("Output shape:", y.shape)   # (1024, 8192)

    # Compare with the original PyTorch implementation (for debug only)
    # import torch.nn.functional as F
    # y_ref = F.linear(x, w, b)
    # y_ref = F.mish(y_ref)
    # y_ref = F.mish(y_ref)
    # print("Max absolute difference:", (y - y_ref).abs().max().item())
