# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_151319/code_3.py
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

# ----------------------------------------------------------------------
# CUDA source – kernels + host launchers
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ----------------------------------------------------------------------
// 1. GEMM with bias (tiled 16×16)
// ----------------------------------------------------------------------
__global__ void gemm_bias_kernel(
    const float* __restrict__ A,   // M×K  (batch × in_features)
    const float* __restrict__ B,   // K×N  (in_features × out_features) – already transposed
    const float* __restrict__ bias,// N
    float* __restrict__ C,         // M×N  (batch × out_features)
    int M, int N, int K)
{
    const int BM = 16;
    const int BN = 16;
    const int BK = 16;
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    int row = blockIdx.y * BM + threadIdx.y;
    int col = blockIdx.x * BN + threadIdx.x;

    float Csub = 0.0f;
    for (int k = 0; k < K; k += BK) {
        // load A tile
        int aRow = row;
        int aCol = k + threadIdx.x;
        if (aRow < M && aCol < K) {
            As[threadIdx.y * BK + threadIdx.x] = A[aRow * K + aCol];
        } else {
            As[threadIdx.y * BK + threadIdx.x] = 0.0f;
        }

        // load B tile
        int bRow = k + threadIdx.y;
        int bCol = col;
        if (bRow < K && bCol < N) {
            Bs[threadIdx.y * BN + threadIdx.x] = B[bRow * N + bCol];
        } else {
            Bs[threadIdx.y * BN + threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // compute partial dot
        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            Csub += As[threadIdx.y * BK + i] * Bs[i * BN + threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = Csub + bias[col];
    }
}

void gemm_bias_launcher(
    at::Tensor A,      // (M, K)
    at::Tensor B,      // (K, N) – transposed weight
    at::Tensor bias,   // (N)
    at::Tensor C)      // (M, N) – output
{
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const int BM = 16, BN = 16;
    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    gemm_bias_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(),
        bias.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K);
}

// ----------------------------------------------------------------------
// 2. Fused normalization + add + mul (one block per batch element)
// ----------------------------------------------------------------------
__global__ void fused_norm_kernel(
    const float* __restrict__ input,   // (B, O)
    const float* __restrict__ y,       // (B, O)
    const float* __restrict__ running_mean, // (O) – may be null if use_input_stats==true
    const float* __restrict__ running_var,  // (O)
    const float* __restrict__ weight,  // (O)
    const float* __restrict__ bias,    // (O)
    bool use_input_stats,
    float eps,
    float* __restrict__ output,        // (B, O)
    int B, int O)
{
    const int blockSize = 256;
    int row = blockIdx.x;                 // one block per batch element
    if (row >= B) return;

    __shared__ float sdata[blockSize * 2]; // [0] = sum, [blockSize] = sum_sq

    int tid = threadIdx.x;
    float sum = 0.0f, sum_sq = 0.0f;

    // ------------------------------------------------------------------
    // 1) Compute per‑row statistics if we must use input statistics
    // ------------------------------------------------------------------
    if (use_input_stats) {
        // partial reduction
        for (int col = tid; col < O; col += blockSize) {
            float v = input[row * O + col];
            sum     += v;
            sum_sq  += v * v;
        }
        sdata[tid]         = sum;
        sdata[tid + blockSize] = sum_sq;
        __syncthreads();

        // tree‑reduction
        for (int s = blockSize / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid]         += sdata[tid + s];
                sdata[tid + blockSize] += sdata[tid + blockSize + s];
            }
            __syncthreads();
        }

        // mean & inverse std in thread 0
        if (tid == 0) {
            float mean = sdata[0] / O;
            float mean_sq = sdata[blockSize] / O;
            float var = mean_sq - mean * mean;
            if (var < 0.0f) var = 0.0f;
            sdata[0]         = mean;
            sdata[blockSize] = rsqrtf(var + eps);
        }
        __syncthreads();
    }

    // ------------------------------------------------------------------
    // 2) Element‑wise pass: normalise, scale, add y, multiply by y
    // ------------------------------------------------------------------
    for (int col = tid; col < O; col += blockSize) {
        float val = input[row * O + col];
        float mean, inv_std;

        if (use_input_stats) {
            mean     = sdata[0];
            inv_std  = sdata[blockSize];
        } else {
            mean    = running_mean[col];
            inv_std = rsqrtf(running_var[col] + eps);
        }

        // instance norm
        float normalized = (val - mean) * inv_std;
        float scaled = normalized * weight[col] + bias[col];

        // add & multiply by y
        float y_val = y[row * O + col];
        float result = (scaled + y_val) * y_val;

        output[row * O + col] = result;
    }
}

void fused_norm_launcher(
    at::Tensor input,           // (B, O)
    at::Tensor y,               // (B, O)
    at::Tensor running_mean,    // (O) – may be empty tensor if not used
    at::Tensor running_var,     // (O)
    at::Tensor weight,          // (O)
    at::Tensor bias,            // (O)
    bool use_input_stats,
    float eps,
    at::Tensor output)          // (B, O)
{
    int B = input.size(0);
    int O = input.size(1);
    const int blockSize = 256;
    dim3 grid(B);
    dim3 block(blockSize);

    fused_norm_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        y.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        use_input_stats,
        eps,
        output.data_ptr<float>(),
        B, O);
}
"""

# ----------------------------------------------------------------------
# C++ binding (PYBIND11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void gemm_bias_launcher(
    at::Tensor A, at::Tensor B,
    at::Tensor bias, at::Tensor C);

void fused_norm_launcher(
    at::Tensor input, at::Tensor y,
    at::Tensor running_mean, at::Tensor running_var,
    at::Tensor weight, at::Tensor bias,
    bool use_input_stats, float eps,
    at::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_bias", &gemm_bias_launcher, "GEMM with bias");
    m.def("fused_norm", &fused_norm_launcher,
          "Fused instance norm + add + mul");
}
"""

# ----------------------------------------------------------------------
# Compile the inline CUDA extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# ----------------------------------------------------------------------
# Functional model – uses the two fused kernels only
# ----------------------------------------------------------------------
def functional_model(
    x,
    y,
    *,
    bmm_weight,
    bmm_bias,
    instance_norm_running_mean,
    instance_norm_running_var,
    instance_norm_weight,
    instance_norm_bias,
    instance_norm_use_input_stats,
    instance_norm_momentum,   # <-- not needed for inference
    instance_norm_eps,
):
    # ------------------------------------------------------------------
    # 1) Make sure all tensors are contiguous CUDA tensors
    # ------------------------------------------------------------------
    x = x.contiguous().cuda()
    y = y.contiguous().cuda()
    bmm_weight = bmm_weight.contiguous().cuda()
    bmm_bias   = bmm_bias.contiguous().cuda()

    # Transpose the weight because our GEMM expects K×N (in‑features × out‑features)
    weight_T = bmm_weight.t().contiguous()      # (in_features, out_features)

    # ------------------------------------------------------------------
    # 2) GEMM + bias  ->  out = x @ weight_T + bias
    # ------------------------------------------------------------------
    B = x.size(0)          # batch size
    O = bmm_weight.size(0) # out_features
    out = torch.empty((B, O), dtype=x.dtype, device=x.device)

    fused_ext.gemm_bias(x, weight_T, bmm_bias, out)

    # ------------------------------------------------------------------
    # 3) Prepare auxiliary tensors for the fused norm kernel
    # ------------------------------------------------------------------
    running_mean = instance_norm_running_mean.contiguous().cuda()
    running_var  = instance_norm_running_var.contiguous().cuda()
    w_norm = instance_norm_weight.contiguous().cuda()
    b_norm = instance_norm_bias.contiguous().cuda()

    # ------------------------------------------------------------------
    # 4) fused_norm: instance‑norm + (x + y) * y
    # ------------------------------------------------------------------
    result = torch.empty_like(out)
    fused_ext.fused_norm(
        out, y,
        running_mean, running_var,
        w_norm, b_norm,
        instance_norm_use_input_stats,
        instance_norm_eps,
        result)

    return result
