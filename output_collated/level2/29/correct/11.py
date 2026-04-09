# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_104842/code_3.py
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

# -------------------------------------------------------------------------
# CUDA source – tiled GEMM + fused double‑mish
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// ---------- device functions ----------
// Mish activation: x * tanh(softplus(x))
__device__ __forceinline__ float mish(float x) {
    // softplus = log(1 + exp(x))
    float sp = logf(1.0f + __expf(x));
    return x * tanhf(sp);
}

// ---------- GEMM + double mish kernel ----------
__global__ void gemm_mish2_kernel(
    const float* __restrict__ A,   // input  (M x K)
    const float* __restrict__ B,   // weights (N x K)
    const float* __restrict__ bias,// bias    (N) or nullptr
    float* __restrict__ C,         // output  (M x N)
    int M, int N, int K)           // sizes
{
    // 16×16 tile size
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    int row = blockIdx.y * 16 + threadIdx.y; // output row (batch)
    int col = blockIdx.x * 16 + threadIdx.x; // output column (out‑feature)

    float sum = 0.0f;

    // loop over K in tiles of 16
    for (int kk = 0; kk < K; kk += 16) {
        // ---- load tile from A (input) ----
        int aRow = row;
        int aCol = kk + threadIdx.x;
        if (aRow < M && aCol < K) {
            As[threadIdx.y][threadIdx.x] = A[aRow * K + aCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // ---- load tile from B (weights) ----
        // B is stored as (N, K) row‑major, we need B[j][k] = B[j*K + k]
        int bRow = kk + threadIdx.y; // k index
        int bCol = col;               // j index (output feature)
        if (bRow < K && bCol < N) {
            Bs[threadIdx.y][threadIdx.x] = B[bCol * K + bRow];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // ---- compute partial dot product for this tile ----
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // ---- write result + bias + double mish ----
    if (row < M && col < N) {
        float val = sum;
        if (bias != nullptr) {
            val += bias[col];
        }
        float m1 = mish(val);
        float m2 = mish(m1);
        C[row * N + col] = m2;
    }
}

// ---------- host wrapper ----------
void gemm_mish2(
    at::Tensor A,
    at::Tensor B,
    at::optional<at::Tensor> bias,
    at::Tensor C)
{
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    A = A.contiguous();
    B = B.contiguous();
    if (bias.has_value()) {
        bias = bias->contiguous();
    }
    C = C.contiguous();

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;
    float* C_ptr = C.data_ptr<float>();

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    gemm_mish2_kernel<<<grid, block>>>(A_ptr, B_ptr, bias_ptr, C_ptr, M, N, K);
    cudaDeviceSynchronize();
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void gemm_mish2(
    at::Tensor A,
    at::Tensor B,
    at::optional<at::Tensor> bias,
    at::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_mish2", &gemm_mish2,
          "Fused GEMM (linear) followed by two mish activations");
}
"""

# -------------------------------------------------------------------------
# Compile the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# The functional model required by the task
# -------------------------------------------------------------------------
def functional_model(
    x,
    *,
    linear_weight,
    linear_bias,
):
    """
    Computes:  y = mish(mish(x @ W^T + b))
    All operations are performed on the GPU using a single fused CUDA kernel.
    """
    # Ensure inputs are on the GPU and contiguous
    if not x.is_cuda:
        x = x.cuda()
    if not linear_weight.is_cuda:
        linear_weight = linear_weight.cuda()
    if linear_bias is not None and not linear_bias.is_cuda:
        linear_bias = linear_bias.cuda()

    x = x.contiguous()
    linear_weight = linear_weight.contiguous()
    if linear_bias is not None:
        linear_bias = linear_bias.contiguous()

    # Allocate output tensor
    batch_size = x.size(0)
    out_features = linear_weight.size(0)
    out = torch.empty((batch_size, out_features),
                      dtype=x.dtype, device=x.device)

    # Call the fused kernel (linear + two mish)
    fused_ext.gemm_mish2(x, linear_weight, linear_bias, out)
    return out

# -------------------------------------------------------------------------
# Helpers for the benchmarking harness (not part of the evaluated code)
# -------------------------------------------------------------------------
def get_init_inputs():
    return [8192, 8192]

def get_inputs():
    return [torch.rand(1024, 8192, device='cuda')]
