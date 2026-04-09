# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_111334/code_3.py
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
# CUDA source – fused linear + mish + mish kernel (tiled GEMM)
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// ----- device function: mish(x) = x * tanh(log(1+exp(x))) -----
__device__ float mish(float x) {
    // softplus = log(1+exp(x))
    float sp = logf(1.0f + expf(x));
    return x * tanhf(sp);
}

// ------------------- fused kernel -------------------------------
__global__ void fused_linear_mish2_kernel(
    const float* __restrict__ A,   // M x K  (row‑major)
    const float* __restrict__ B,   // K x N  (row‑major, transposed weight)
    const float* __restrict__ bias,// N
    float* __restrict__ C,         // M x N output
    int M, int N, int K)
{
    // Shared memory for tiles
    __shared__ float s_A[TILE_M][TILE_K];
    __shared__ float s_B[TILE_K][TILE_N];

    // Global indices for this thread
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    float acc = 0.0f;

    // Loop over K in tiles
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // ---- load tile from A (M x K) ----
        int aCol = k_tile + threadIdx.x;
        if (row < M && aCol < K) {
            s_A[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // ---- load tile from B (K x N) ----
        int bRow = k_tile + threadIdx.y;
        if (bRow < K && col < N) {
            s_B[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // ---- compute partial dot product for this tile ----
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            acc += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    // ---- apply bias, then two mish activations ----
    if (row < M && col < N) {
        acc += bias[col];
        float m1 = mish(acc);
        float m2 = mish(m1);
        C[row * N + col] = m2;
    }
}

void launch_fused_op(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& bias,
    torch::Tensor& C) {
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    dim3 block(TILE_N, TILE_M);

    fused_linear_mish2_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        bias.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
}
"""

# ----------------------------------------------------------------------
# C++ host code – binding for the CUDA kernel
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void launch_fused_op(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& bias,
    torch::Tensor& C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_fused_op,
          "Fused linear + mish + mish kernel");
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
# Boilerplate that matches the original benchmark interface
# ----------------------------------------------------------------------
batch_size = 1024
in_features = 8192
out_features = 8192

def get_init_inputs():
    return [in_features, out_features]

def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda')]

# ----------------------------------------------------------------------
# Optimized functional_model – fused linear + two mish passes
# ----------------------------------------------------------------------
def functional_model(
    x: torch.Tensor,
    *,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Fused linear layer followed by two mish activations.
    Input:
        x          – (batch, in_features)
        linear_weight – (out_features, in_features)
        linear_bias   – (out_features,)
    Output:
        (batch, out_features)
    """
    # Weight must be transposed to K×N row‑major layout for the kernel.
    weight_t = linear_weight.t().contiguous()          # (in_features, out_features)

    # Allocate output tensor
    y = torch.empty((x.size(0), linear_weight.size(0)),
                    dtype=x.dtype, device=x.device)

    # Launch the fused CUDA kernel
    fused_ext.fused_op(x, weight_t, linear_bias, y)

    return y
