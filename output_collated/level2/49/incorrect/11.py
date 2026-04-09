# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_093635/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'softmax_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, applies Softmax and Sigmoid.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

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
    # State for conv_transpose (nn.ConvTranspose3d)
    if 'conv_transpose_weight' in flat_state:
        state_kwargs['conv_transpose_weight'] = flat_state['conv_transpose_weight']
    else:
        state_kwargs['conv_transpose_weight'] = getattr(model.conv_transpose, 'weight', None)
    if 'conv_transpose_bias' in flat_state:
        state_kwargs['conv_transpose_bias'] = flat_state['conv_transpose_bias']
    else:
        state_kwargs['conv_transpose_bias'] = getattr(model.conv_transpose, 'bias', None)
    state_kwargs['conv_transpose_stride'] = model.conv_transpose.stride
    state_kwargs['conv_transpose_padding'] = model.conv_transpose.padding
    state_kwargs['conv_transpose_output_padding'] = model.conv_transpose.output_padding
    state_kwargs['conv_transpose_groups'] = model.conv_transpose.groups
    state_kwargs['conv_transpose_dilation'] = model.conv_transpose.dilation
    # State for softmax (nn.Softmax)
    state_kwargs['softmax_dim'] = model.softmax.dim
    # State for sigmoid (nn.Sigmoid)
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
#  CUDA source – two kernels and their host wrappers
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// ---------------------------------------------------------------------
// 1) Naïve transposed 3‑D convolution (groups == 1)
// ---------------------------------------------------------------------
__global__ void conv_transpose3d_fwd_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int K, const int stride, const int padding,
    const int dilation, const int /*output_padding*/,
    const int K3)               // K³ = K*K*K
{
    const int total_out = N * C_out * D_out * H_out * W_out;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    // unravel flat index to (n,co,d,h,w)
    int tmp = idx;
    int w = tmp % W_out; tmp /= W_out;
    int h = tmp % H_out; tmp /= H_out;
    int d = tmp % D_out; tmp /= D_out;
    int co = tmp % C_out; tmp /= C_out;
    int n = tmp;

    float sum = (bias != nullptr) ? bias[co] : 0.0f;

    // naive loops – groups are assumed to be 1
    for (int ci = 0; ci < C_in; ++ci) {
        for (int kd = 0; kd < K; ++kd) {
            int den_d = d - padding + kd * dilation;
            if (den_d % stride != 0) continue;
            int di = den_d / stride;
            if (di < 0 || di >= D_in) continue;

            for (int kh = 0; kh < K; ++kh) {
                int den_h = h - padding + kh * dilation;
                if (den_h % stride != 0) continue;
                int hi = den_h / stride;
                if (hi < 0 || hi >= H_in) continue;

                for (int kw = 0; kw < K; ++kw) {
                    int den_w = w - padding + kw * dilation;
                    if (den_w % stride != 0) continue;
                    int wi = den_w / stride;
                    if (wi < 0 || wi >= W_in) continue;

                    // weight index: (ci*C_out + co) * K³ + (kd*K + kh)*K + kw
                    int wIdx = ((ci * C_out + co) * K3 + (kd * K + kh) * K + kw);
                    float wval = weight[wIdx];

                    // input index (flattened)
                    int iIdx = ((n * C_in + ci) * D_in + di) * (H_in * W_in) + (hi * W_in + wi);
                    float ival = input[iIdx];

                    sum += ival * wval;
                }
            }
        }
    }
    output[idx] = sum;
}

// ---------------------------------------------------------------------
// 2) Fused softmax + sigmoid (any reduction dimension)
// ---------------------------------------------------------------------
__global__ void fused_softmax_sigmoid_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N, const int C, const int D, const int H, const int W,
    const int softmax_dim,          // 0‑N,1‑C,2‑D,3‑H,4‑W
    const int dim_size,             // size of the dimension we reduce over
    const int slice_stride)         // stride between consecutive elements along that dim
{
    extern __shared__ float sdata[];    // size = dim_size
    const int tid = threadIdx.x;
    const int slice_id = blockIdx.x;    // one block per (N,D,H,W) or (N,C,H,W) … slice
    const int base = slice_id * slice_stride;

    // load value
    float val = input[base + tid * slice_stride];
    sdata[tid] = val;
    __syncthreads();

    // ----- max reduction (for numerical stability) -----
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + offset]);
        __syncthreads();
    }
    float maxval = sdata[0];

    // exponentiate
    float expval = expf(val - maxval);
    sdata[tid] = expval;
    __syncthreads();

    // ----- sum reduction -----
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset)
            sdata[tid] += sdata[tid + offset];
        __syncthreads();
    }
    float sum = sdata[0];

    // ----- softmax → sigmoid -----
    float softmax = expval / (sum + 1e-8f);
    float out = 1.0f / (1.0f + expf(-softmax));

    output[base + tid * slice_stride] = out;
}

// ---------------------------------------------------------------------
// Host wrappers – called from Python
// ---------------------------------------------------------------------
void conv_transpose3d_fwd(int N, int C_in, int C_out,
                          int D_in, int H_in, int W_in,
                          int D_out, int H_out, int W_out,
                          int K, int stride, int padding,
                          int dilation, int output_padding,
                          const float* input,
                          const float* weight,
                          const float* bias,
                          float* output)
{
    const int K3 = K * K * K;
    const int total_out = N * C_out * D_out * H_out * W_out;
    const int BLOCK = 256;
    const int grid = (total_out + BLOCK - 1) / BLOCK;
    conv_transpose3d_fwd_kernel<<<grid, BLOCK>>>(
        input, weight, bias, output,
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        K, stride, padding, dilation, output_padding,
        K3);
    cudaDeviceSynchronize();
}

void fused_softmax_sigmoid(int N, int C, int D, int H, int W,
                           int softmax_dim,
                           const float* input,
                           float* output)
{
    // determine size of the dimension we reduce over and the stride to the next element
    int dim_size = 0;
    int slice_stride = 0;
    if (softmax_dim == 0) {            // reduce over N
        dim_size = N;
        slice_stride = C * D * H * W;
    } else if (softmax_dim == 1) {     // reduce over C
        dim_size = C;
        slice_stride = D * H * W;
    } else if (softmax_dim == 2) {     // reduce over D
        dim_size = D;
        slice_stride = H * W;
    } else if (softmax_dim == 3) {     // reduce over H
        dim_size = H;
        slice_stride = W;
    } else {                           // reduce over W
        dim_size = W;
        slice_stride = 1;
    }

    const int blocks = (N * C * D * H * W) / dim_size;   // one block per slice
    const int shmem = dim_size * sizeof(float);
    fused_softmax_sigmoid_kernel<<<blocks, dim_size, shmem>>>(
        input, output,
        N, C, D, H, W,
        softmax_dim, dim_size, slice_stride);
    cudaDeviceSynchronize();
}
"""

# -------------------------------------------------------------------------
#  C++ interface – pybind11 bindings
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void conv_transpose3d_fwd(int N, int C_in, int C_out,
                          int D_in, int H_in, int W_in,
                          int D_out, int H_out, int W_out,
                          int K, int stride, int padding,
                          int dilation, int output_padding,
                          const float* input,
                          const float* weight,
                          const float* bias,
                          float* output);

void fused_softmax_sigmoid(int N, int C, int D, int H, int W,
                           int softmax_dim,
                           const float* input,
                           float* output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose3d_fwd", &conv_transpose3d_fwd,
          "Transposed 3‑D convolution (naïve CUDA)");
    m.def("fused_softmax_sigmoid", &fused_softmax_sigmoid,
          "Fused softmax + sigmoid on arbitrary dimension");
}
"""

# -------------------------------------------------------------------------
#  Compile the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# -------------------------------------------------------------------------
#  Functional model that will be imported by the evaluator
# -------------------------------------------------------------------------
def functional_model(
    x: torch.Tensor,
    *,
    conv_transpose_weight: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
    conv_transpose_stride: int,
    conv_transpose_padding: int,
    conv_transpose_output_padding: int,
    conv_transpose_groups: int,   # not used – we assume groups==1
    conv_transpose_dilation: int,
    softmax_dim: int,
) -> torch.Tensor:
    """
    Custom implementation:
      1) Naïve transposed 3‑D convolution (no built‑in convolution).
      2) Fused softmax + sigmoid on the requested dimension.

    The only extra kernel launch is the conv kernel; the element‑wise
    part is performed in a single fused kernel, which eliminates a
    whole global‑memory read/write pass and reduces launch overhead.
    """

    # ------------------------------------------------------------
    # Move data to GPU if needed
    # ------------------------------------------------------------
    if not x.is_cuda:
        x = x.cuda()
    if not conv_transpose_weight.is_cuda:
        conv_transpose_weight = conv_transpose_weight.cuda()
    # Ensure a bias tensor exists – if not, use a zero tensor (won't affect result)
    if conv_transpose_bias is None:
        conv_transpose_bias = torch.zeros(conv_transpose_weight.shape[1],
                                          dtype=x.dtype, device='cuda')
    else:
        if not conv_transpose_bias.is_cuda:
            conv_transpose_bias = conv_transpose_bias.cuda()

    # ------------------------------------------------------------
    # Compute output shapes (same formula as PyTorch)
    # ------------------------------------------------------------
    N, C_in, D_in, H_in, W_in = x.shape
    K = conv_transpose_weight.shape[2]          # square kernel
    C_out = conv_transpose_weight.shape[1]      # out‑channels (groups==1)
    stride = conv_transpose_stride
    padding = conv_transpose_padding
    dilation = conv_transpose_dilation
    output_padding = conv_transpose_output_padding

    D_out = (D_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1

    # ------------------------------------------------------------
    # 1) Transposed convolution
    # ------------------------------------------------------------
    conv_out = torch.empty(N, C_out, D_out, H_out, W_out,
                          dtype=x.dtype, device='cuda')
    fused_ext.conv_transpose3d_fwd(
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        K, stride, padding, dilation, output_padding,
        x.data_ptr(),
        conv_transpose_weight.data_ptr(),
        conv_transpose_bias.data_ptr(),
        conv_out.data_ptr())

    # ------------------------------------------------------------
    # 2) Fused softmax + sigmoid
    # ------------------------------------------------------------
    final_out = torch.empty_like(conv_out)
    fused_ext.fused_softmax_sigmoid(
        N, C_out, D_out, H_out, W_out,
        softmax_dim,
        conv_out.data_ptr(),
        final_out.data_ptr())

    return final_out


# -----------------------------------------------------------------
# Helpers required by the harness (not used by the evaluator)
# -----------------------------------------------------------------
def get_init_inputs():
    return [32, 64, 3, 2, 1, 1]   # in_ch, out_ch, k, s, p, op


def get_inputs():
    return [torch.rand(16, 32, 16, 32, 32, dtype=torch.float32)]
