# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150223/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'eps', 'momentum']
FORWARD_ARG_NAMES = ['x', 'y']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['bmm_weight', 'bmm_bias', 'instance_norm_running_mean', 'instance_norm_running_var', 'instance_norm_weight', 'instance_norm_bias', 'instance_norm_use_input_stats', 'instance_norm_momentum', 'instance_norm_eps']
REQUIRED_FLAT_STATE_NAMES = ['bmm_weight', 'bmm_bias', 'instance_norm_running_mean', 'instance_norm_running_var', 'instance_norm_weight', 'instance_norm_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a batch matrix multiplication, instance normalization, summation, residual addition, and multiplication.
    """

    def __init__(self, in_features, out_features, eps=1e-05, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)

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
    # State for bmm (nn.Linear)
    if 'bmm_weight' in flat_state:
        state_kwargs['bmm_weight'] = flat_state['bmm_weight']
    else:
        state_kwargs['bmm_weight'] = getattr(model.bmm, 'weight', None)
    if 'bmm_bias' in flat_state:
        state_kwargs['bmm_bias'] = flat_state['bmm_bias']
    else:
        state_kwargs['bmm_bias'] = getattr(model.bmm, 'bias', None)
    # State for instance_norm (nn.InstanceNorm2d)
    if 'instance_norm_running_mean' in flat_state:
        state_kwargs['instance_norm_running_mean'] = flat_state['instance_norm_running_mean']
    else:
        state_kwargs['instance_norm_running_mean'] = getattr(model.instance_norm, 'running_mean', None)
    if 'instance_norm_running_var' in flat_state:
        state_kwargs['instance_norm_running_var'] = flat_state['instance_norm_running_var']
    else:
        state_kwargs['instance_norm_running_var'] = getattr(model.instance_norm, 'running_var', None)
    if 'instance_norm_weight' in flat_state:
        state_kwargs['instance_norm_weight'] = flat_state['instance_norm_weight']
    else:
        state_kwargs['instance_norm_weight'] = getattr(model.instance_norm, 'weight', None)
    if 'instance_norm_bias' in flat_state:
        state_kwargs['instance_norm_bias'] = flat_state['instance_norm_bias']
    else:
        state_kwargs['instance_norm_bias'] = getattr(model.instance_norm, 'bias', None)
    state_kwargs['instance_norm_use_input_stats'] = not model.instance_norm.track_running_stats
    state_kwargs['instance_norm_momentum'] = model.instance_norm.momentum
    state_kwargs['instance_norm_eps'] = model.instance_norm.eps
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
# CUDA source – two tiled kernels plus wrapper functions
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// ---------------------------------------------------------------------
// Tiled GEMM kernel:  C = A @ B  (A: N×Cin, B: Cin×Cout)
// ---------------------------------------------------------------------
__global__ void gemm_tiled_kernel(
    const float* __restrict__ A,   // N × Cin
    const float* __restrict__ B,   // Cin × Cout  (transposed weight)
    const float* __restrict__ bias,// Cout
    float* __restrict__ C,         // N × Cout
    int N, int Cin, int Cout)
{
    const int TILE = 16;
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;   // batch index
    int col = blockIdx.x * TILE + threadIdx.x;   // output feature

    float acc = 0.0f;

    // loop over the Cin dimension in tiles
    for (int t = 0; t < (Cin + TILE - 1) / TILE; ++t) {
        // load tile from A
        int aCol = t * TILE + threadIdx.x;
        if (row < N && aCol < Cin) {
            As[threadIdx.y][threadIdx.x] = A[row * Cin + aCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // load tile from B (B is stored transposed: Cin × Cout)
        int bRow = t * TILE + threadIdx.y;
        if (bRow < Cin && col < Cout) {
            Bs[threadIdx.y][threadIdx.x] = B[bRow * Cout + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // dot product for the tile
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < Cout) {
        float out = acc;
        if (bias != nullptr) out += bias[col];
        C[row * Cout + col] = out;
    }
}

// ---------------------------------------------------------------------
// Fused instance‑norm + element‑wise ( + y, * y ) kernel
// ---------------------------------------------------------------------
__global__ void instance_norm_fused_kernel(
    const float* __restrict__ input,   // N × Cout  (linear output)
    const float* __restrict__ weight,  // Cout
    const float* __restrict__ bias,    // Cout
    const float* __restrict__ y,       // N × Cout
    float* __restrict__ output,        // N × Cout
    int N, int Cout,
    float eps)
{
    const int BLOCK = 256;
    int n = blockIdx.x;          // batch index
    int tid = threadIdx.x;

    // -------- per‑thread partial sums ----------
    float sum = 0.0f, sum_sq = 0.0f;
    for (int c = tid; c < Cout; c += BLOCK) {
        float v = input[n * Cout + c];
        sum    += v;
        sum_sq += v * v;
    }

    // -------- block‑level reduction ----------
    __shared__ float sdata[2 * BLOCK];
    sdata[tid]         = sum;
    sdata[tid + BLOCK] = sum_sq;
    __syncthreads();

    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid]         += sdata[tid + s];
            sdata[tid + BLOCK] += sdata[tid + BLOCK + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sum    = sdata[0];
        sum_sq = sdata[BLOCK];
    }
    __syncthreads();

    float mean   = sum / (float)Cout;
    float var    = sum_sq / (float)Cout - mean * mean;
    var = fmaxf(var, 0.0f);
    float inv_std = rsqrtf(var + eps);

    // -------- final normalization + element‑wise ops ----------
    for (int c = tid; c < Cout; c += BLOCK) {
        float v   = input[n * Cout + c];
        float nrm = (v - mean) * inv_std;
        float w   = weight[c];
        float b   = bias[c];
        float yv  = y[n * Cout + c];

        float out = nrm * w + b;   // instance‑norm scale & shift
        out = out + yv;             // + y
        out = out * yv;             // * y

        output[n * Cout + c] = out;
    }
}

// ---------------------------------------------------------------------
// Host wrappers (called from Python)
// ---------------------------------------------------------------------
void gemm_tiled(int N, int Cin, int Cout,
                const float* A, const float* B,
                const float* bias, float* C) {
    dim3 block(16, 16);
    dim3 grid((Cout + 15) / 16, (N + 15) / 16);
    gemm_tiled_kernel<<<grid, block>>>(A, B, bias, C, N, Cin, Cout);
    cudaDeviceSynchronize();
}

void instance_norm_fused(int N, int Cout,
                         const float* input,
                         const float* weight,
                         const float* bias,
                         const float* y,
                         float* output,
                         float eps) {
    dim3 block(256);
    dim3 grid(N);                     // one block per batch element
    instance_norm_fused_kernel<<<grid, block>>>(input, weight, bias, y,
                                                 output, N, Cout, eps);
    cudaDeviceSynchronize();
}
"""

# -------------------------------------------------------------------------
# C++ binding – exposes the two host functions to Python
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void gemm_tiled(int N, int Cin, int Cout,
                const float* A, const float* B,
                const float* bias, float* C);
void instance_norm_fused(int N, int Cout,
                         const float* input,
                         const float* weight,
                         const float* bias,
                         const float* y,
                         float* output,
                         float eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_tiled", &gemm_tiled, "Tiled GEMM kernel");
    m.def("instance_norm_fused", &instance_norm_fused,
          "Fused instance norm + element‑wise ops");
}
"""

# -------------------------------------------------------------------------
# Build the inline CUDA extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# -------------------------------------------------------------------------
# The functional model that will be evaluated
# -------------------------------------------------------------------------
def functional_model(
    x,
    y,
    *,
    bmm_weight,
    bmm_bias,
    instance_norm_running_mean,      # ignored (only input stats are used)
    instance_norm_running_var,       # ignored
    instance_norm_weight,
    instance_norm_bias,
    instance_norm_use_input_stats,   # ignored (we always use input stats)
    instance_norm_momentum,          # ignored
    instance_norm_eps,
):
    # -----------------------------------------------------------------
    # Ensure all tensors are float32 on the GPU
    # -----------------------------------------------------------------
    x   = x.float().cuda()
    y   = y.float().cuda()
    bmm_weight = bmm_weight.float().cuda()
    bmm_bias   = bmm_bias.float().cuda()
    w_norm     = instance_norm_weight.float().cuda()
    b_norm     = instance_norm_bias.float().cuda()

    # -----------------------------------------------------------------
    # Prepare transposed weight matrix (Cin × Cout) for the tiled GEMM
    # -----------------------------------------------------------------
    weight_t = bmm_weight.t().contiguous()   # (Cin × Cout)

    N, Cin   = x.shape
    Cout     = bmm_weight.shape[0]

    # -----------------------------------------------------------------
    # 1) Tiled matrix‑multiplication (linear layer)
    # -----------------------------------------------------------------
    linear_out = torch.empty((N, Cout), dtype=torch.float32, device='cuda')
    fused_ext.gemm_tiled(
        N, Cin, Cout,
        x.data_ptr(),
        weight_t.data_ptr(),
        bmm_bias.data_ptr(),
        linear_out.data_ptr()
    )

    # -----------------------------------------------------------------
    # 2) Fused instance‑norm + (+y) + (*y)
    # -----------------------------------------------------------------
    out = torch.empty_like(y)
    fused_ext.instance_norm_fused(
        N, Cout,
        linear_out.data_ptr(),
        w_norm.data_ptr(),
        b_norm.data_ptr(),
        y.data_ptr(),
        out.data_ptr(),
        float(instance_norm_eps)
    )

    return out


# -------------------------------------------------------------------------
# Helper functions required by the evaluation harness
# -------------------------------------------------------------------------
def get_init_inputs():
    in_features  = 8192
    out_features = 8192
    return [in_features, out_features]


def get_inputs():
    batch_size = 1024
    in_features  = 8192
    out_features = 8192
    return [
        torch.rand(batch_size, in_features),
        torch.rand(batch_size, out_features)
    ]
