# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_152808/code_6.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'kernel_size', 'scale_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['matmul_weight', 'matmul_bias', 'max_pool_kernel_size', 'max_pool_stride', 'max_pool_padding', 'max_pool_dilation', 'max_pool_ceil_mode', 'max_pool_return_indices', 'scale_factor']
REQUIRED_FLAT_STATE_NAMES = ['matmul_weight', 'matmul_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs matrix multiplication, max pooling, sum, and scaling.
    """

    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.max_pool = nn.MaxPool1d(kernel_size)
        self.scale_factor = scale_factor

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
    # State for matmul (nn.Linear)
    if 'matmul_weight' in flat_state:
        state_kwargs['matmul_weight'] = flat_state['matmul_weight']
    else:
        state_kwargs['matmul_weight'] = getattr(model.matmul, 'weight', None)
    if 'matmul_bias' in flat_state:
        state_kwargs['matmul_bias'] = flat_state['matmul_bias']
    else:
        state_kwargs['matmul_bias'] = getattr(model.matmul, 'bias', None)
    # State for max_pool (nn.MaxPool1d)
    state_kwargs['max_pool_kernel_size'] = model.max_pool.kernel_size
    state_kwargs['max_pool_stride'] = model.max_pool.stride
    state_kwargs['max_pool_padding'] = model.max_pool.padding
    state_kwargs['max_pool_dilation'] = model.max_pool.dilation
    state_kwargs['max_pool_ceil_mode'] = model.max_pool.ceil_mode
    state_kwargs['max_pool_return_indices'] = model.max_pool.return_indices
    if 'scale_factor' in flat_state:
        state_kwargs['scale_factor'] = flat_state['scale_factor']
    else:
        state_kwargs['scale_factor'] = getattr(model, 'scale_factor')
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
# CUDA kernels – tiled matrix‑vector product + fused max‑pool/sum/scale
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ----- parameters ------------------------------------------------------------
constexpr int BLOCK   = 256;          // threads per block (multiple of 32)
constexpr int TILE    = 32;           // tile size for the inner dimension

// ----- linear kernel:  y = X @ W^T + bias  (tiled GEMV) --------------------
__global__ void linear_kernel(
    const float* __restrict__ x,      // (B, I)
    const float* __restrict__ weight, // (O, I)
    const float* __restrict__ bias,   // (O) or nullptr
    float* __restrict__ y,            // (B, O) output of linear part
    const int B, const int I, const int O)
{
    const int b   = blockIdx.x;                 // batch index
    const int k0  = blockIdx.y * BLOCK;         // first output index for this block
    const int tid = threadIdx.x;
    const int k   = k0 + tid;
    if (k >= O) return;

    const float* x_ptr  = x  + b * I;
    const float* w_row  = weight + k * I;

    float sum = (bias != nullptr) ? bias[k] : 0.0f;

    // shared memory:  BLOCK * TILE  weight tile  +  TILE  input tile
    __shared__ float sdata[BLOCK * TILE + TILE];
    float* weight_tile = sdata;
    float* x_tile      = sdata + BLOCK * TILE;

    for (int iStart = 0; iStart < I; iStart += TILE) {
        // ---- load a tile of the input vector (first TILE threads) ----
        if (tid < TILE) {
            int idx = iStart + tid;
            x_tile[tid] = (idx < I) ? x_ptr[idx] : 0.0f;
        }
        __syncthreads();

        // ---- load the corresponding slice of the weight row ----------
        // each thread loads TILE elements of its own row
        #pragma unroll TILE
        for (int i = 0; i < TILE; ++i) {
            int idx = iStart + i;
            weight_tile[tid * TILE + i] = (idx < I) ? w_row[idx] : 0.0f;
        }
        __syncthreads();

        // ---- compute partial dot product -------------------------------
        #pragma unroll TILE
        for (int i = 0; i < TILE; ++i) {
            sum += weight_tile[tid * TILE + i] * x_tile[i];
        }
        __syncthreads();
    }

    y[b * O + k] = sum;
}

// ----- pool‑sum kernel: max‑pool + sum * scale (block‑wise reduction) -----
__global__ void pool_sum_kernel(
    const float* __restrict__ y,   // (B, O) linear output
    float* __restrict__ out,       // (B) final result
    const int B, const int O,
    const int K, const int S,
    const int padding, const int dilation,
    const bool ceil_mode,
    const float scale,
    const int L_out)               // number of pooling windows
{
    const int b   = blockIdx.x;                     // batch
    const int win_base = blockIdx.y * BLOCK;       // first window handled by this block
    const int tid = threadIdx.x;
    const int win = win_base + tid;                // global window index

    // compute max of the current window
    float win_max = -1e38f;
    if (win < L_out) {
        int start = win * S - padding;
        for (int i = 0; i < K; ++i) {
            int idx = start + i * dilation;
            if (0 <= idx && idx < O) {
                float v = y[b * O + idx];
                if (v > win_max) win_max = v;
            }
        }
    } else {
        win_max = 0.0f;   // dummy, will not be added
    }

    // ---- block‑wise sum of window maxima --------------------------------
    __shared__ float sdata[BLOCK];
    sdata[tid] = win_max;
    __syncthreads();

    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&out[b], sdata[0] * scale);
    }
}

// ----- host‑side wrapper ----------------------------------------------------
torch::Tensor fused_op(
    torch::Tensor x,          // (B, I)
    torch::Tensor weight,    // (O, I)
    torch::Tensor bias,      // (O) or empty
    int K, int S,
    int padding, int dilation,
    bool ceil_mode,
    float scale)
{
    const int B = x.size(0);
    const int I = x.size(1);
    const int O = weight.size(0);

    // ---------- 1) linear part ----------
    auto y = torch::empty({B, O}, x.options());

    int grid_y = (O + BLOCK - 1) / BLOCK;
    dim3 grid_lin(B, grid_y);
    linear_kernel<<<grid_lin, BLOCK,
                    (BLOCK * TILE + TILE) * sizeof(float)>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        y.data_ptr<float>(),
        B, I, O);

    // ---------- 2) pooling + sum + scale ----------
    // number of output windows (same formula as PyTorch's max_pool1d)
    int L_out;
    if (!ceil_mode) {
        L_out = (O + 2 * padding - dilation * (K - 1) - 1) / S + 1;
    } else {
        L_out = (O + 2 * padding - dilation * (K - 1) - 1 + S - 1) / S + 1;
    }

    auto out = torch::zeros({B}, x.options());

    int grid_y2 = (L_out + BLOCK - 1) / BLOCK;
    dim3 grid_pool(B, grid_y2);
    pool_sum_kernel<<<grid_pool, BLOCK>>>(
        y.data_ptr<float>(),
        out.data_ptr<float>(),
        B, O, K, S, padding, dilation, ceil_mode, scale, L_out);

    return out;
}
"""

# -------------------------------------------------------------------------
# C++ binding (pybind11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

torch::Tensor fused_op(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int K, int S,
    int padding, int dilation,
    bool ceil_mode,
    float scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op,
          "Fused linear + max‑pool + sum + scale (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("K"), py::arg("S"),
          py::arg("padding"), py::arg("dilation"),
          py::arg("ceil_mode"), py::arg("scale"));
}
"""

# -------------------------------------------------------------------------
# Build the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Public functional model – will be imported and evaluated
# -------------------------------------------------------------------------
def functional_model(
    x,                      # (B, I)
    *,                      # keyword‑only arguments
    matmul_weight,          # (O, I)
    matmul_bias,            # (O) or None
    max_pool_kernel_size,   # K
    max_pool_stride,        # S
    max_pool_padding=0,       # padding
    max_pool_dilation=1,      # dilation
    max_pool_ceil_mode=False,     # bool
    max_pool_return_indices=False, # ignored (not needed)
    scale_factor=1.0            # float
):
    """
    Fused CUDA implementation:
        1) Linear:  y = x @ W^T + b
        2) Max‑pool1d over the output dimension
        3) Sum of the pooled values
        4) Multiply by scale_factor
    All steps are performed by custom kernels, avoiding any
    torch.mm / torch.nn.functional.* calls.
    """
    # matmul_bias may be None; we pass an empty tensor in that case
    bias = matmul_bias if matmul_bias is not None else torch.empty(0, device='cuda')

    out = fused_ext.fused_op(
        x,
        matmul_weight,
        bias,
        max_pool_kernel_size,
        max_pool_stride,
        max_pool_padding,
        max_pool_dilation,
        max_pool_ceil_mode,
        scale_factor
    )
    return out
