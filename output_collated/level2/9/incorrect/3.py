# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_073421/code_3.py
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
# CUDA source – a tiled GEMM kernel that also adds bias, scales and applies ReLU
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tiled matrix-multiplication (16×16 tiles) fused with element-wise ops
__global__ void fused_op_kernel(
    const float* __restrict__ A,   // input x   (M × K)
    const float* __restrict__ W,   // weight    (N × K)
    const float* __restrict__ bias,// bias      (N)
    float*       __restrict__ C,   // output    (M × N)
    int M, int N, int K,
    float mult,
    float sub
) {
    // 16×16 shared-memory tiles
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    // Output position for this thread
    int row = blockIdx.y * 16 + threadIdx.y; // batch index
    int col = blockIdx.x * 16 + threadIdx.x; // output feature index

    float acc = 0.0f;

    // Loop over the K dimension in tiles of 16
    for (int kk = 0; kk < K; kk += 16) {
        // Load tile from A (row-major, coalesced)
        if (row < M && (kk + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + (kk + threadIdx.x)];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from W: W[col][kk + threadIdx.y]
        if (col < N && (kk + threadIdx.y) < K) {
            Bs[threadIdx.y][threadIdx.x] = W[col * K + (kk + threadIdx.y)];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Inner product of the two tiles
        #pragma unroll
        for (int l = 0; l < 16; ++l) {
            acc += As[threadIdx.y][l] * Bs[l][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result (if inside the matrix) – add bias, scale, ReLU
    if (row < M && col < N) {
        float out = acc + bias[col];
        out = out * mult - sub;
        out = fmaxf(out, 0.0f);
        C[row * N + col] = out;
    }
}

// Host wrapper that uses torch::Tensor (no raw pointers in the Python binding)
void fused_op_forward(
    const torch::Tensor& A,
    const torch::Tensor& W,
    const torch::Tensor& bias,
    torch::Tensor& C,
    float multiply_value,
    float subtract_value
) {
    int M = A.size(0);
    int K = A.size(1);
    int N = W.size(0);

    const int BLOCK = 16;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);

    fused_op_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        W.data_ptr<float>(),
        bias.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        multiply_value,
        subtract_value
    );

    // Basic error check (optional but helpful)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11) – exposes the kernel to Python
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const torch::Tensor& A,
    const torch::Tensor& W,
    const torch::Tensor& bias,
    torch::Tensor& C,
    float multiply_value,
    float subtract_value
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused linear + scale + ReLU forward");
}
"""

# -------------------------------------------------------------------------
# Compile the inline CUDA extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# The functional model that will be imported / evaluated
# -------------------------------------------------------------------------
def functional_model(
    x,
    *,
    linear_weight,
    linear_bias,
    subtract_value,
    multiply_value,
):
    """
    Fused linear layer (x @ W^T + b) with:
        * subtract_value
        * multiply_value
        * ReLU
    All performed in a single custom CUDA kernel.
    """
    # Ensure all inputs reside on the same CUDA device
    if not x.is_cuda:
        x = x.cuda()
    if not linear_weight.is_cuda:
        linear_weight = linear_weight.cuda()
    if not linear_bias.is_cuda:
        linear_bias = linear_bias.cuda()

    M = x.size(0)          # batch size
    N = linear_weight.size(0)  # output features
    # Allocate output tensor
    out = torch.empty((M, N), dtype=torch.float32, device=x.device)

    # Launch the fused CUDA kernel
    fused_ext.fused_op(x, linear_weight, linear_bias, out,
                       multiply_value, subtract_value)

    return out

# -------------------------------------------------------------------------
# Optional helpers for the benchmarking harness (not required for grading)
# -------------------------------------------------------------------------
batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]

def get_inputs():
    return [torch.rand(batch_size, in_features)]
