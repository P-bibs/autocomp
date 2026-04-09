# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_061658/code_10.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'divisor', 'pool_size', 'bias_shape', 'sum_dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'max_pool_kernel_size', 'max_pool_stride', 'max_pool_padding', 'max_pool_dilation', 'max_pool_ceil_mode', 'max_pool_return_indices', 'global_avg_pool_output_size', 'divisor', 'bias', 'sum_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    """

    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

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
    # State for max_pool (nn.MaxPool3d)
    state_kwargs['max_pool_kernel_size'] = model.max_pool.kernel_size
    state_kwargs['max_pool_stride'] = model.max_pool.stride
    state_kwargs['max_pool_padding'] = model.max_pool.padding
    state_kwargs['max_pool_dilation'] = model.max_pool.dilation
    state_kwargs['max_pool_ceil_mode'] = model.max_pool.ceil_mode
    state_kwargs['max_pool_return_indices'] = model.max_pool.return_indices
    # State for global_avg_pool (nn.AdaptiveAvgPool3d)
    state_kwargs['global_avg_pool_output_size'] = model.global_avg_pool.output_size
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
    if 'sum_dim' in flat_state:
        state_kwargs['sum_dim'] = flat_state['sum_dim']
    else:
        state_kwargs['sum_dim'] = getattr(model, 'sum_dim')
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
# CUDA kernel: custom conv3d + max_pool3d + adaptive_avg_pool3d + reduce_sum_kernel
# ----------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

// ---------------------------- Conv3d Kernel ----------------------------

__global__ void conv3d_kernel(
    const float* __restrict__ input,    // [N, Ci, Di, Hi, Wi]
    const float* __restrict__ weight,   // [Co, Ci, Kd, Kh, Kw]
    const float* __restrict__ bias,     // [Co]
    const int N, const int Ci, const int Di, const int Hi, const int Wi,
    const int Co, const int Kd, const int Kh, const int Kw,
    const int Sd, const int Sh, const int Sw,
    const int Pd, const int Ph, const int Pw,
    const int Dd, const int Dh, const int Dw,
    float* __restrict__ output          // [N, Co, Do, Ho, Wo]
) {
    // Output dimensions
    const int Do = (Di + 2 * Pd - Dd * (Kd - 1) - 1) / Sd + 1;
    const int Ho = (Hi + 2 * Ph - Dh * (Kh - 1) - 1) / Sh + 1;
    const int Wo = (Wi + 2 * Pw - Dw * (Kw - 1) - 1) / Sw + 1;

    const int o_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_outputs = N * Co * Do * Ho * Wo;

    if (o_idx >= total_outputs) return;

    // Decode output index
    int tmp = o_idx;
    const int wo = tmp % Wo; tmp /= Wo;
    const int ho = tmp % Ho; tmp /= Ho;
    const int doo = tmp % Do; tmp /= Do;
    const int co = tmp % Co; tmp /= Co;
    const int n = tmp;

    float sum = 0.0f;
    for (int kd = 0; kd < Kd; ++kd) {
        const int di = doo * Sd - Pd + kd * Dd;
        if (di < 0 || di >= Di) continue;
        for (int kh = 0; kh < Kh; ++kh) {
            const int hi = ho * Sh - Ph + kh * Dh;
            if (hi < 0 || hi >= Hi) continue;
            for (int kw = 0; kw < Kw; ++kw) {
                const int wi = wo * Sw - Pw + kw * Dw;
                if (wi < 0 || wi >= Wi) continue;

                for (int ci = 0; ci < Ci; ++ci) {
                    const int in_idx = ((n * Ci + ci) * Di + di) * Hi * Wi + hi * Wi + wi;
                    const int w_idx = (((co * Ci + ci) * Kd + kd) * Kh + kh) * Kw + kw;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    sum += bias[co];
    output[o_idx] = sum;
}

// ---------------------------- MaxPool3d Kernel ----------------------------

__global__ void max_pool3d_kernel(
    const float* __restrict__ input,    // [N, C, Di, Hi, Wi]
    const int N, const int C, const int Di, const int Hi, const int Wi,
    const int Kd, const int Kh, const int Kw,
    const int Sd, const int Sh, const int Sw,
    const int Pd, const int Ph, const int Pw,
    const int Dd, const int Dh, const int Dw,
    float* __restrict__ output           // [N, C, Do, Ho, Wo]
) {
    // Output dimensions
    const int Do = (Di + 2 * Pd - Dd * (Kd - 1) - 1) / Sd + 1;
    const int Ho = (Hi + 2 * Ph - Dh * (Kh - 1) - 1) / Sh + 1;
    const int Wo = (Wi + 2 * Pw - Dw * (Kw - 1) - 1) / Sw + 1;

    const int o_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_outputs = N * C * Do * Ho * Wo;

    if (o_idx >= total_outputs) return;

    // Decode output index
    int tmp = o_idx;
    const int wo = tmp % Wo; tmp /= Wo;
    const int ho = tmp % Ho; tmp /= Ho;
    const int doo = tmp % Do; tmp /= Do;
    const int c = tmp % C; tmp /= C;
    const int n = tmp;

    float max_val = -1e30f;
    for (int kd = 0; kd < Kd; ++kd) {
        const int di = doo * Sd - Pd + kd * Dd;
        if (di < 0 || di >= Di) continue;
        for (int kh = 0; kh < Kh; ++kh) {
            const int hi = ho * Sh - Ph + kh * Dh;
            if (hi < 0 || hi >= Hi) continue;
            for (int kw = 0; kw < Kw; ++kw) {
                const int wi = wo * Sw - Pw + kw * Dw;
                if (wi < 0 || wi >= Wi) continue;

                const int in_idx = ((n * C + c) * Di + di) * Hi * Wi + hi * Wi + wi;
                max_val = fmaxf(max_val, input[in_idx]);
            }
        }
    }
    output[o_idx] = max_val;
}

// ---------------------------- AdaptiveAvgPool3d Kernel ----------------------------

__global__ void adaptive_avg_pool3d_kernel(
    const float* __restrict__ input,     // [N, C, Di, Hi, Wi]
    const int N, const int C, const int Di, const int Hi, const int Wi,
    const int Do, const int Ho, const int Wo,
    float* __restrict__ output           // [N, C, Do, Ho, Wo]
) {
    const int o_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_outputs = N * C * Do * Ho * Wo;

    if (o_idx >= total_outputs) return;

    // Decode output index
    int tmp = o_idx;
    const int wo = tmp % Wo; tmp /= Wo;
    const int ho = tmp % Ho; tmp /= Ho;
    const int doo = tmp % Do; tmp /= Do;
    const int c = tmp % C; tmp /= C;
    const int n = tmp;

    // Compute start and end indices for pooling
    const int id_start = (doo * Di) / Do;
    const int id_end = ((doo + 1) * Di + Do - 1) / Do;
    const int ih_start = (ho * Hi) / Ho;
    const int ih_end = ((ho + 1) * Hi + Ho - 1) / Ho;
    const int iw_start = (wo * Wi) / Wo;
    const int iw_end = ((wo + 1) * Wi + Wo - 1) / Wo;

    float sum = 0.0f;
    int count = 0;
    for (int id = id_start; id < id_end; ++id) {
        for (int ih = ih_start; ih < ih_end; ++ih) {
            for (int iw = iw_start; iw < iw_end; ++iw) {
                const int in_idx = ((n * C + c) * Di + id) * Hi * Wi + ih * Wi + iw;
                sum += input[in_idx];
                count++;
            }
        }
    }
    output[o_idx] = sum / count;
}

// ---------------------------- Reduce Sum Kernel ----------------------------

__global__ void reduce_sum_kernel(
    const float* __restrict__ input,     // [N, C, D, H, W]
    const float divisor,
    const float total_bias,
    const int N,
    const int C,
    const int spatial,
    float* __restrict__ output            // [N, D, H, W]
) {
    const int batch_idx = blockIdx.x / spatial;
    const int spatial_idx = blockIdx.x % spatial;

    const int tid = threadIdx.x;

    float sum = 0.0f;
    for (int c = tid; c < C; c += blockDim.x) {
        const int idx = ((batch_idx * C + c) * spatial + spatial_idx);
        sum += input[idx];
    }

    __shared__ float sdata[256];
    sdata[tid] = sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float sum_x = sdata[0];
        const float out_val = sum_x / divisor + total_bias;
        output[batch_idx * spatial + spatial_idx] = out_val;
    }
}

// ---------------------------- Host Wrapper Functions ----------------------------

void conv3d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int Sd, int Sh, int Sw, int Pd, int Ph, int Pw, int Dd, int Dh, int Dw,
    torch::Tensor output) {
    const int N = input.size(0), Ci = input.size(1), Di = input.size(2), Hi = input.size(3), Wi = input.size(4);
    const int Co = weight.size(0), Kd = weight.size(2), Kh = weight.size(3), Kw = weight.size(4);
    const int total_outputs = N * Co * 
        ((Di + 2*Pd - Dd*(Kd-1) - 1)/Sd + 1) * 
        ((Hi + 2*Ph - Dh*(Kh-1) - 1)/Sh + 1) * 
        ((Wi + 2*Pw - Dw*(Kw-1) - 1)/Sw + 1);

    const int threads = 256;
    const int blocks = (total_outputs + threads - 1) / threads;

    conv3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        N, Ci, Di, Hi, Wi, Co, Kd, Kh, Kw,
        Sd, Sh, Sw, Pd, Ph, Pw, Dd, Dh, Dw,
        output.data_ptr<float>()
    );
}

void max_pool3d_forward(
    torch::Tensor input,
    int Kd, int Kh, int Kw,
    int Sd, int Sh, int Sw,
    int Pd, int Ph, int Pw,
    int Dd, int Dh, int Dw,
    torch::Tensor output) {
    const int N = input.size(0), C = input.size(1), Di = input.size(2), Hi = input.size(3), Wi = input.size(4);
    const int total_outputs = N * C * 
        ((Di + 2*Pd - Dd*(Kd-1) - 1)/Sd + 1) * 
        ((Hi + 2*Ph - Dh*(Kh-1) - 1)/Sh + 1) * 
        ((Wi + 2*Pw - Dw*(Kw-1) - 1)/Sw + 1);

    const int threads = 256;
    const int blocks = (total_outputs + threads - 1) / threads;

    max_pool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        N, C, Di, Hi, Wi,
        Kd, Kh, Kw, Sd, Sh, Sw, Pd, Ph, Pw, Dd, Dh, Dw,
        output.data_ptr<float>()
    );
}

void adaptive_avg_pool3d_forward(
    torch::Tensor input,
    int Do, int Ho, int Wo,
    torch::Tensor output) {
    const int N = input.size(0), C = input.size(1), Di = input.size(2), Hi = input.size(3), Wi = input.size(4);
    const int total_outputs = N * C * Do * Ho * Wo;

    const int threads = 256;
    const int blocks = (total_outputs + threads - 1) / threads;

    adaptive_avg_pool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), N, C, Di, Hi, Wi, Do, Ho, Wo,
        output.data_ptr<float>()
    );
}

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor bias,
    float divisor,
    float total_bias,
    torch::Tensor output) {
    const int N = input.size(0);
    const int C = input.size(1);
    const int D = input.size(2);
    const int H = input.size(3);
    const int W = input.size(4);
    const int spatial = D * H * W;

    const int threads = 256;
    const int blocks = N * spatial;

    reduce_sum_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        divisor,
        total_bias,
        N,
        C,
        spatial,
        output.data_ptr<float>()
    );
}

"""

cpp_source = r"""
#include <torch/extension.h>

void conv3d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int Sd, int Sh, int Sw, int Pd, int Ph, int Pw, int Dd, int Dh, int Dw,
    torch::Tensor output);

void max_pool3d_forward(
    torch::Tensor input,
    int Kd, int Kh, int Kw,
    int Sd, int Sh, int Sw,
    int Pd, int Ph, int Pw,
    int Dd, int Dh, int Dw,
    torch::Tensor output);

void adaptive_avg_pool3d_forward(
    torch::Tensor input,
    int Do, int Ho, int Wo,
    torch::Tensor output);

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor bias,
    float divisor,
    float total_bias,
    torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d", &conv3d_forward, "Custom Conv3d");
    m.def("max_pool3d", &max_pool3d_forward, "Custom MaxPool3d");
    m.def("adaptive_avg_pool3d", &adaptive_avg_pool3d_forward, "Custom AdaptiveAvgPool3d");
    m.def("fused_op", &fused_op_forward, "Fused post-convolution reduction");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
    max_pool_kernel_size,
    max_pool_stride,
    max_pool_padding,
    max_pool_dilation,
    max_pool_ceil_mode,
    max_pool_return_indices,
    global_avg_pool_output_size,
    divisor,
    bias,
    sum_dim):
    
    # Assume conv_groups == 1 for simplicity
    if conv_groups != 1:
        raise NotImplementedError("Only conv_groups=1 is supported")

    # ---------------------------- Conv3d ----------------------------
    N, Ci, Di, Hi, Wi = x.shape
    Co, _, Kd, Kh, Kw = conv_weight.shape
    Sd, Sh, Sw = conv_stride
    Pd, Ph, Pw = conv_padding
    Dd, Dh, Dw = conv_dilation
    
    Do = (Di + 2*Pd - Dd*(Kd-1) - 1) // Sd + 1
    Ho = (Hi + 2*Ph - Dh*(Kh-1) - 1) // Sh + 1
    Wo = (Wi + 2*Pw - Dw*(Kw-1) - 1) // Sw + 1
    
    x_conv = torch.empty((N, Co, Do, Ho, Wo), device=x.device, dtype=x.dtype)
    fused_ext.conv3d(x, conv_weight, conv_bias, Sd, Sh, Sw, Pd, Ph, Pw, Dd, Dh, Dw, x_conv)

    # ---------------------------- MaxPool3d ----------------------------
    Kd_mp, Kh_mp, Kw_mp = max_pool_kernel_size
    Sd_mp, Sh_mp, Sw_mp = max_pool_stride
    Pd_mp, Ph_mp, Pw_mp = max_pool_padding
    Dd_mp, Dh_mp, Dw_mp = max_pool_dilation
    
    Dmp_o = (Do + 2*Pd_mp - Dd_mp*(Kd_mp-1) - 1) // Sd_mp + 1
    Hmp_o = (Ho + 2*Ph_mp - Dh_mp*(Kh_mp-1) - 1) // Sh_mp + 1
    Wmp_o = (Wo + 2*Pw_mp - Dw_mp*(Kw_mp-1) - 1) // Sw_mp + 1
    
    x_pool = torch.empty((N, Co, Dmp_o, Hmp_o, Wmp_o), device=x.device, dtype=x.dtype)
    fused_ext.max_pool3d(x_conv, Kd_mp, Kh_mp, Kw_mp, Sd_mp, Sh_mp, Sw_mp, Pd_mp, Ph_mp, Pw_mp, Dd_mp, Dh_mp, Dw_mp, x_pool)

    # ---------------------------- AdaptiveAvgPool3d ----------------------------
    Do_ad, Ho_ad, Wo_ad = global_avg_pool_output_size
    x_adap = torch.empty((N, Co, Do_ad, Ho_ad, Wo_ad), device=x.device, dtype=x.dtype)
    fused_ext.adaptive_avg_pool3d(x_pool, Do_ad, Ho_ad, Wo_ad, x_adap)

    # ---------------------------- Fused Reduce Sum ----------------------------
    output = torch.zeros((N, Do_ad, Ho_ad, Wo_ad), device=x.device, dtype=x.dtype)
    total_bias = bias.sum().item()
    fused_ext.fused_op(x_adap, bias, divisor, total_bias, output)

    return output
