# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_000325/code_10.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'num_groups']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'group_norm_weight', 'group_norm_bias', 'group_norm_num_groups', 'group_norm_eps']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'group_norm_weight', 'group_norm_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D convolution, applies Group Normalization, computes the mean
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

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
    # State for conv (nn.Conv3d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
    # State for group_norm (nn.GroupNorm)
    if 'group_norm_weight' in flat_state:
        state_kwargs['group_norm_weight'] = flat_state['group_norm_weight']
    else:
        state_kwargs['group_norm_weight'] = getattr(model.group_norm, 'weight', None)
    if 'group_norm_bias' in flat_state:
        state_kwargs['group_norm_bias'] = flat_state['group_norm_bias']
    else:
        state_kwargs['group_norm_bias'] = getattr(model.group_norm, 'bias', None)
    state_kwargs['group_norm_num_groups'] = model.group_norm.num_groups
    state_kwargs['group_norm_eps'] = model.group_norm.eps
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
# CUDA source – contains both the naive 3‑D convolution and the fused
# group‑normalisation + global‑reduction kernel that uses grid‑stride loops
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// -------------------------------------------------
// 3‑D convolution – one thread per output element
// -------------------------------------------------
__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,          // may be nullptr
    float* __restrict__ output,
    int N, int inC, int inD, int inH, int inW,
    int outC, int outD, int outH, int outW,
    int kD, int kH, int kW,
    int sD, int sH, int sW,
    int pD, int pH, int pW,
    int dD, int dH, int dW,
    int groups)
{
    int outPerSample = outC * outD * outH * outW;
    int totalOut = N * outPerSample;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalOut) return;

    int n       = tid / outPerSample;
    int rem     = tid % outPerSample;
    int oc      = rem / (outD * outH * outW);
    rem         = rem % (outD * outH * outW);
    int od      = rem / (outH * outW);
    rem         = rem % (outH * outW);
    int oh      = rem / outW;
    int ow      = rem % outW;

    int inC_per_g = inC / groups;
    int outC_per_g = outC / groups;
    int g = oc / outC_per_g;                     // group index
    int inC_start = g * inC_per_g;

    float sum = 0.0f;

    for (int kd = 0; kd < kD; ++kd) {
        int iD = od * sD - pD + kd * dD;
        if (iD < 0 || iD >= inD) continue;
        for (int kh = 0; kh < kH; ++kh) {
            int iH = oh * sH - pH + kh * dH;
            if (iH < 0 || iH >= inH) continue;
            for (int kw = 0; kw < kW; ++kw) {
                int iW = ow * sW - pW + kw * dW;
                if (iW < 0 || iW >= inW) continue;
                for (int ic = 0; ic < inC_per_g; ++ic) {
                    int ic_global = inC_start + ic;
                    // weight index: (oc * inC_per_g + ic) * (kD*kH*kW) + …
                    int wIdx = ((oc * inC_per_g + ic) * kD + kd) * (kH * kW) + (kh * kW + kw);
                    // input index
                    int iIdx = (((n * inC + ic_global) * inD + iD) * inH + iH) * inW + iW;
                    sum += input[iIdx] * weight[wIdx];
                }
            }
        }
    }
    if (bias) sum += bias[oc];
    int oIdx = (((n * outC + oc) * outD + od) * outH + oh) * outW + ow;
    output[oIdx] = sum;
}

void conv3d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output,
    int N, int inC, int inD, int inH, int inW,
    int outC, int outD, int outH, int outW,
    int kD, int kH, int kW,
    int sD, int sH, int sW,
    int pD, int pH, int pW,
    int dD, int dH, int dW,
    int groups)
{
    const int threads = 256;
    int outPerSample = outC * outD * outH * outW;
    int total = N * outPerSample;
    int blocks = (total + threads - 1) / threads;
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    conv3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
        output.data_ptr<float>(),
        N, inC, inD, inH, inW,
        outC, outD, outH, outW,
        kD, kH, kW,
        sD, sH, sW,
        pD, pH, pW,
        dD, dH, dW,
        groups);
}

// -------------------------------------------------
// Fused Group‑Norm + Global‑Reduction kernel
//   – Uses grid‑stride loops inside each block
// -------------------------------------------------
__global__ void fused_norm_reduce_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ group_sums,          // size N * G
    int N, int C, int D, int H, int W, int G, float eps)
{
    // One block per (batch, group)
    int ng = blockIdx.x;                     // linearised index n*G + g
    int n = ng / G;
    int g = ng % G;

    int C_per_G = C / G;
    int spatial = D * H * W;
    int group_size = C_per_G * spatial;
    int c_start = g * C_per_G;

    // shared memory: first blockDim for sum, second blockDim for sum_sq
    extern __shared__ float smem[];
    float* s_sum   = smem;
    float* s_sum2  = smem + blockDim.x;

    // -----------------------------------------------------------------
    // Pass 1: compute sum(x) and sum(x^2) for the whole group
    // -----------------------------------------------------------------
    float sum  = 0.0f;
    float sum2 = 0.0f;
    for (int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        int local_c = idx / spatial;
        int spatial_idx = idx % spatial;
        int c = c_start + local_c;

        int w = spatial_idx % W;
        int tmp = spatial_idx / W;
        int h = tmp % H;
        int d = tmp / H;

        int offset = (((n * C + c) * D + d) * H + h) * W + w;
        float x = input[offset];
        sum  += x;
        sum2 += x * x;
    }
    s_sum[threadIdx.x]  = sum;
    s_sum2[threadIdx.x] = sum2;
    __syncthreads();

    // block‑wide parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x]  += s_sum[threadIdx.x + stride];
            s_sum2[threadIdx.x] += s_sum2[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // compute mean, variance, inv_std in thread 0
    if (threadIdx.x == 0) {
        float mean  = s_sum[0]  / (float)group_size;
        float var   = s_sum2[0] / (float)group_size - mean * mean;
        if (var < eps) var = eps;
        float inv_std = rsqrtf(var);
        s_sum[0]  = mean;      // reuse s_sum[0] for mean
        s_sum2[0] = inv_std;   // reuse s_sum2[0] for inv_std
    }
    __syncthreads();

    float mean    = s_sum[0];
    float inv_std = s_sum2[0];

    // -----------------------------------------------------------------
    // Pass 2: compute sum of (normalised * weight + bias)
    // -----------------------------------------------------------------
    float sum_y = 0.0f;
    for (int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        int local_c = idx / spatial;
        int spatial_idx = idx % spatial;
        int c = c_start + local_c;

        int w = spatial_idx % W;
        int tmp = spatial_idx / W;
        int h = tmp % H;
        int d = tmp / H;

        int offset = (((n * C + c) * D + d) * H + h) * W + w;
        float x   = input[offset];
        float x_norm = (x - mean) * inv_std;
        float y = x_norm * weight[c] + bias[c];
        sum_y += y;
    }
    s_sum[threadIdx.x] = sum_y;
    __syncthreads();

    // final reduction of sum_y
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        group_sums[ng] = s_sum[0];
    }
}

void fused_norm_reduce_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor group_sums,
    int N, int C, int D, int H, int W, int G, float eps)
{
    const int threads = 256;
    int blocks = N * G;                         // one block per (batch, group)
    int smem = 2 * threads * sizeof(float);    // sum + sum_sq
    fused_norm_reduce_kernel<<<blocks, threads, smem>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        group_sums.data_ptr<float>(),
        N, C, D, H, W, G, eps);
}
"""

# ----------------------------------------------------------------------
# C++ interface – bindings for the two CUDA functions
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void conv3d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output,
    int N, int inC, int inD, int inH, int inW,
    int outC, int outD, int outH, int outW,
    int kD, int kH, int kW,
    int sD, int sH, int sW,
    int pD, int pH, int pW,
    int dD, int dH, int dW,
    int groups);

void fused_norm_reduce_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor group_sums,
    int N, int C, int D, int H, int W, int G, float eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d_forward", &conv3d_forward, "3‑D convolution forward");
    m.def("fused_norm_reduce_forward", &fused_norm_reduce_forward,
          "Fused GroupNorm + global reduction forward");
}
"""

# ----------------------------------------------------------------------
# Compile the inline CUDA extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# The functional model that will be imported / evaluated
# ----------------------------------------------------------------------
def functional_model(
    x, *,
    conv_weight, conv_bias,
    conv_stride, conv_padding, conv_dilation, conv_groups,
    group_norm_weight, group_norm_bias,
    group_norm_num_groups, group_norm_eps,
):
    # -------------------------------------------------
    # 1) Custom 3‑D convolution (no torch.nn.functional)
    # -------------------------------------------------
    N  = x.size(0)
    inC = x.size(1)
    inD = x.size(2)
    inH = x.size(3)
    inW = x.size(4)

    kD, kH, kW = conv_weight.shape[2], conv_weight.shape[3], conv_weight.shape[4]
    outC = conv_weight.size(0)

    # output spatial size
    outD = (inD + 2*conv_padding[0] - conv_dilation[0]*(kD-1) - 1)//conv_stride[0] + 1
    outH = (inH + 2*conv_padding[1] - conv_dilation[1]*(kH-1) - 1)//conv_stride[1] + 1
    outW = (inW + 2*conv_padding[2] - conv_dilation[2]*(kW-1) - 1)//conv_stride[2] + 1

    conv_out = torch.empty((N, outC, outD, outH, outW), dtype=x.dtype, device=x.device)

    # optional bias
    bias_tensor = conv_bias if conv_bias is not None else torch.empty(0, dtype=x.dtype, device=x.device)

    fused_ext.conv3d_forward(
        x, conv_weight, bias_tensor, conv_out,
        N, inC, inD, inH, inW,
        outC, outD, outH, outW,
        kD, kH, kW,
        conv_stride[0], conv_stride[1], conv_stride[2],
        conv_padding[0], conv_padding[1], conv_padding[2],
        conv_dilation[0], conv_dilation[1], conv_dilation[2],
        conv_groups
    )

    # -------------------------------------------------
    # 2) Fused group‑normalisation + global reduction
    #    – uses the grid‑stride‑loop kernel
    # -------------------------------------------------
    C = outC
    D = outD
    H = outH
    W = outW
    G = group_norm_num_groups

    # make sure weight / bias are on the right device
    w_norm = group_norm_weight
    b_norm = group_norm_bias if group_norm_bias is not None else torch.zeros(C, dtype=x.dtype, device=x.device)

    group_sums = torch.empty(N * G, dtype=torch.float32, device=x.device)

    fused_ext.fused_norm_reduce_forward(
        conv_out, w_norm, b_norm, group_sums,
        N, C, D, H, W, G, group_norm_eps
    )

    # -------------------------------------------------
    # 3) Final global mean (simple reduction on the host side)
    # -------------------------------------------------
    # reshape to (batch, groups) and sum over groups
    total_elements = C * D * H * W
    out = group_sums.view(N, G).sum(dim=1).float() / float(total_elements)
    return out
