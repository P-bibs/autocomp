# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_075823/code_3.py
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
# CUDA kernel + host function (fused linear + subtract + multiply + ReLU)
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tile sizes – these can be tuned but 32×32 works well on RTX 2080 Ti
constexpr int BM = 32;   // output rows per block (batch dimension)
constexpr int BN = 32;   // output columns per block (out_features)
constexpr int BK = 32;   // inner tile size (in_features)

__global__ void fused_op_kernel(
    const float* __restrict__ x,      // (M, K)
    const float* __restrict__ w,      // (N, K)
    const float* __restrict__ bias,   // (N)
    float* __restrict__ out,          // (M, N)
    const float subtract_value,
    const float multiply_value,
    const int M,                      // batch size
    const int K,                      // in_features
    const int N)                      // out_features
{
    // Shared memory for the two tiles
    extern __shared__ float s[];
    float* A = s;               // tile of x   – size BM * BK
    float* B = s + BM * BK;     // tile of w   – size BK * BN

    // Output position for this thread
    const int row = blockIdx.y * BM + threadIdx.y;   // batch index
    const int col = blockIdx.x * BN + threadIdx.x;   // out_feature index

    // Accumulator for the dot product
    float sum = 0.0f;

    // Loop over the K dimension in BK-sized tiles
    for (int k = 0; k < K; k += BK) {
        // ---- load tile A (x) ----
        int x_col = k + threadIdx.x;
        if (row < M && x_col < K) {
            A[threadIdx.y * BK + threadIdx.x] = x[row * K + x_col];
        } else {
            A[threadIdx.y * BK + threadIdx.x] = 0.0f;
        }

        // ---- load tile B (weight) ----
        // w is stored row-major: w[out][in] -> index = out*K + in
        int w_row = k + threadIdx.y;          // in-feature inside the tile
        int w_col = col;                      // out-feature (constant for the block)
        if (w_row < K && w_col < N) {
            B[threadIdx.y * BN + threadIdx.x] = w[w_col * K + w_row];
        } else {
            B[threadIdx.y * BN + threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // ---- compute partial dot product ----
        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            sum += A[threadIdx.y * BK + i] * B[i * BN + threadIdx.x];
        }

        __syncthreads();
    }

    // ---- finalise one output element ----
    if (row < M && col < N) {
        float val = sum + bias[col];                // add bias
        val = (val - subtract_value) * multiply_value; // subtract & multiply
        val = fmaxf(val, 0.0f);                     // ReLU
        out[row * N + col] = val;
    }
}

void fused_op_forward(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor bias,
    torch::Tensor out,
    float subtract_value,
    float multiply_value)
{
    const int M = x.size(0);      // batch size
    const int K = x.size(1);      // in_features
    const int N = w.size(0);      // out_features

    dim3 block(BN, BM);                       // 32×32 threads per block
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    const int shared_mem = (BM * BK + BK * BN) * sizeof(float);
    fused_op_kernel<<<grid, block, shared_mem>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        subtract_value,
        multiply_value,
        M, K, N);
    cudaDeviceSynchronize();
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor bias,
    torch::Tensor out,
    float subtract_value,
    float multiply_value);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward,
          "Fused linear + subtract + multiply + ReLU kernel");
}
"""

# -------------------------------------------------------------------------
# Compile the inline extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
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
    Fused CUDA kernel:
        y = x @ linear_weight.T + linear_bias
        y = (y - subtract_value) * multiply_value
        y = relu(y)
    """
    # Make sure tensors are on the GPU
    if not x.is_cuda:
        x = x.cuda()
    if not linear_weight.is_cuda:
        linear_weight = linear_weight.cuda()
    if not linear_bias.is_cuda:
        linear_bias = linear_bias.cuda()

    batch_size = x.size(0)
    out_features = linear_weight.size(0)

    # Allocate output (batch_size, out_features)
    out = torch.empty((batch_size, out_features),
                      dtype=x.dtype, device=x.device)

    # Invoke the custom fused kernel
    fused_ext.fused_op(
        x,
        linear_weight,
        linear_bias,
        out,
        float(subtract_value),
        float(multiply_value),
    )
    return out

# -------------------------------------------------------------------------
# Helper functions required by the original benchmark harness
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
