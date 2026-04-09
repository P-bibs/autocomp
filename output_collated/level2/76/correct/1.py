# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_014309/code_3.py
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
# Inline CUDA source – fused linear + bias + ReLU kernel
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

constexpr int BM = 16;   // block tile height (batch dimension)
constexpr int BN = 16;   // block tile width  (output features)
constexpr int BK = 16;   // inner tile size (reduction dimension)

template <typename scalar_t>
__global__ void fused_op_kernel(
    const scalar_t* __restrict__ x,       // (batch, in_features)
    const scalar_t* __restrict__ weight,  // (out_features, in_features)
    const scalar_t* __restrict__ bias,    // (out_features)
    scalar_t* __restrict__ output,        // (batch, out_features)
    const int batch_size,
    const int in_features,
    const int out_features)
{
    // Shared memory for two tiles: A (BM x BK) and B (BK x BN)
    extern __shared__ char smem[];
    scalar_t* As = (scalar_t*)smem;
    scalar_t* Bs = (scalar_t*)(smem + BM * BK * sizeof(scalar_t));

    const int by = blockIdx.y;   // tile index in batch direction
    const int bx = blockIdx.x;   // tile index in output-feature direction
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int row = by * BM + ty;   // global row (batch index)
    const int col = bx * BN + tx;   // global column (output feature)

    scalar_t sum = 0;

    // --- tiled matrix-multiply loop ---------------------------------------
    for (int k = 0; k < in_features; k += BK) {
        // Load tile from x (BM x BK) – coalesced read
        if (row < batch_size && (k + tx) < in_features) {
            As[ty * BK + tx] = x[row * in_features + (k + tx)];
        } else {
            As[ty * BK + tx] = 0;
        }

        // Load tile from weight (BK x BN) – transposed for coalescing
        if (col < out_features && (k + ty) < in_features) {
            Bs[ty * BN + tx] = weight[col * in_features + (k + ty)];
        } else {
            Bs[ty * BN + tx] = 0;
        }

        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            sum += As[ty * BK + i] * Bs[i * BN + tx];
        }

        __syncthreads();
    }

    // --- bias addition + ReLU (fused) ------------------------------------
    if (row < batch_size && col < out_features) {
        sum += bias[col];
        sum = (sum > 0) ? sum : static_cast<scalar_t>(0);   // ReLU
        output[row * out_features + col] = sum;
    }
}

// Host function that launches the kernel
void fused_op_forward(
    at::Tensor x,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor output)
{
    const int batch_size = x.size(0);
    const int in_features = x.size(1);
    const int out_features = weight.size(0);

    const int block_m = BM;
    const int block_n = BN;
    const int grid_m = (batch_size + block_m - 1) / block_m;
    const int grid_n = (out_features + block_n - 1) / block_n;

    dim3 block(block_m, block_n);
    dim3 grid(grid_n, grid_m);

    const int shared_mem = (BM * BK + BK * BN) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_op_forward", ([&] {
        fused_op_kernel<scalar_t><<<grid, block, shared_mem>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, in_features, out_features);
    }));
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    at::Tensor x,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused linear+bias+relu forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional model – now uses the fused kernel
# -------------------------------------------------------------------------
def functional_model(x, *, gemm_weight, bias):
    """
    Fused linear (GEMM) + bias addition + ReLU.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch, in_features).
    gemm_weight : torch.Tensor
        Weight matrix of shape (out_features, in_features).
    bias : torch.Tensor
        Bias vector of shape (out_features,).

    Returns
    -------
    torch.Tensor
        Output tensor of shape (batch, out_features) after ReLU.
    """
    # Move data to GPU if not already there
    if not x.is_cuda:
        x = x.cuda()
    if not gemm_weight.is_cuda:
        gemm_weight = gemm_weight.cuda()
    if not bias.is_cuda:
        bias = bias.cuda()

    batch_size = x.size(0)
    out_features = gemm_weight.size(0)

    # Allocate output buffer
    output = torch.empty((batch_size, out_features), dtype=x.dtype, device='cuda')

    # Launch the fused CUDA kernel
    fused_ext.fused_op(x, gemm_weight, bias, output)

    return output


# -------------------------------------------------------------------------
# Helper functions (unchanged, required by the harness)
# -------------------------------------------------------------------------
batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)


def get_init_inputs():
    return [in_features, out_features, bias_shape]


def get_inputs():
    return [torch.rand(batch_size, in_features)]
