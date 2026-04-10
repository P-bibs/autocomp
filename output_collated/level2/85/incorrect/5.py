# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_141846/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'num_groups', 'scale_shape', 'maxpool_kernel_size', 'clamp_min', 'clamp_max']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'group_norm_weight', 'group_norm_bias', 'group_norm_num_groups', 'group_norm_eps', 'maxpool_kernel_size', 'maxpool_stride', 'maxpool_padding', 'maxpool_dilation', 'maxpool_ceil_mode', 'maxpool_return_indices', 'scale', 'clamp_min', 'clamp_max']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'group_norm_weight', 'group_norm_bias', 'scale']


class ModelNew(nn.Module):
    """
    ModelNew that performs convolution, group normalization, scaling, max pooling, and clamping.
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

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
    # State for conv (nn.Conv2d)
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
    # State for maxpool (nn.MaxPool2d)
    state_kwargs['maxpool_kernel_size'] = model.maxpool.kernel_size
    state_kwargs['maxpool_stride'] = model.maxpool.stride
    state_kwargs['maxpool_padding'] = model.maxpool.padding
    state_kwargs['maxpool_dilation'] = model.maxpool.dilation
    state_kwargs['maxpool_ceil_mode'] = model.maxpool.ceil_mode
    state_kwargs['maxpool_return_indices'] = model.maxpool.return_indices
    if 'scale' in flat_state:
        state_kwargs['scale'] = flat_state['scale']
    else:
        state_kwargs['scale'] = getattr(model, 'scale')
    if 'clamp_min' in flat_state:
        state_kwargs['clamp_min'] = flat_state['clamp_min']
    else:
        state_kwargs['clamp_min'] = getattr(model, 'clamp_min')
    if 'clamp_max' in flat_state:
        state_kwargs['clamp_max'] = flat_state['clamp_max']
    else:
        state_kwargs['clamp_max'] = getattr(model, 'clamp_max')
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

# Define CUDA kernel for fused operation: Conv + GroupNorm + Scale + MaxPool + Clamp
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Helper function to compute mean and variance for group norm
__device__ void compute_mean_var(const float* __restrict__ x, int N, int C, int H, int W, int group_idx, int num_groups, float gn_eps, float* mean, float* var) {
    int group_size = (C / num_groups) * H * W;
    int start_idx = group_idx * group_size;
    int end_idx = start_idx + group_size;

    float sum = 0.f;
    for (int i = start_idx; i < end_idx; ++i) {
        sum += x[i];
    }
    *mean = sum / group_size;

    float sum_sq = 0.f;
    for (int i = start_idx; i < end_idx; ++i) {
        float diff = x[i] - (*mean);
        sum_sq += diff * diff;
    }
    *var = sum_sq / group_size + gn_eps;
}

// Fused kernel
__global__ void fused_op_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ gn_weight,
    const float* __restrict__ gn_bias,
    float* __restrict__ output,
    int B, int Ci, int Hi, int Wi,
    int Co, int Kh, int Kw,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int gn_groups, float gn_eps,
    int mp_kh, int mp_kw,
    int mp_sh, int mp_sw,
    int mp_ph, int mp_pw,
    float scale,
    float clamp_min,
    float clamp_max
) {
    int batch_idx = blockIdx.z;
    int out_ch = blockIdx.y;
    int out_row = blockIdx.x;
    int tid = threadIdx.x;

    // Shared memory for intermediate results (tile)
    extern __shared__ float shared_mem[];

    if (batch_idx >= B || out_ch >= Co || out_row >= (Hi + 2 * pad_h - Kh) / stride_h + 1) return;

    int Ho = (Hi + 2 * pad_h - Kh) / stride_h + 1;
    int Wo = (Wi + 2 * pad_w - Kw) / stride_w + 1;

    int mp_Ho = (Ho + 2 * mp_ph - mp_kh) / mp_sh + 1;
    int mp_Wo = (Wo + 2 * mp_pw - mp_kw) / mp_sw + 1;

    for (int out_col = 0; out_col < mp_Wo; ++out_col) {
        float max_val = -1e30f;
        bool valid = false;

        // Max pool window
        for (int mp_i = 0; mp_i < mp_kh; ++mp_i) {
            for (int mp_j = 0; mp_j < mp_kw; ++mp_j) {
                int conv_i = out_row * mp_sh - mp_ph + mp_i;
                int conv_j = out_col * mp_sw - mp_pw + mp_j;

                if (conv_i % stride_h != 0 || conv_j % stride_w != 0) continue;
                conv_i /= stride_h;
                conv_j /= stride_w;

                // Convolution
                float conv_result = 0.f;
                if (conv_i >= 0 && conv_i < Ho && conv_j >= 0 && conv_j < Wo) {
                    for (int k_i = 0; k_i < Kh; ++k_i) {
                        for (int k_j = 0; k_j < Kw; ++k_j) {
                            int in_i = conv_i * stride_h - pad_h + k_i;
                            int in_j = conv_j * stride_w - pad_w + k_j;
                            if (in_i >= 0 && in_i < Hi && in_j >= 0 && in_j < Wi) {
                                for (int c = tid; c < Ci; c += blockDim.x) {
                                    int in_idx = batch_idx * (Ci * Hi * Wi) + c * (Hi * Wi) + in_i * Wi + in_j;
                                    int w_idx = out_ch * (Ci * Kh * Kw) + c * (Kh * Kw) + k_i * Kw + k_j;
                                    conv_result += input[in_idx] * conv_weight[w_idx];
                                }
                            }
                        }
                    }
                }
                __syncthreads();

                // Add bias
                if (tid == 0) {
                    conv_result += conv_bias[out_ch];
                }
                __syncthreads();

                // Group Norm (per-group statistics)
                int group_idx = out_ch / (Co / gn_groups);
                float mean, var;
                compute_mean_var(shared_mem, 1, Co, Ho, Wo, group_idx, gn_groups, gn_eps, &mean, &var);
                float inv_std = rsqrtf(var);
                float gn_result = (conv_result - mean) * inv_std;
                gn_result = gn_result * gn_weight[out_ch] + gn_bias[out_ch];

                // Scale
                float scaled = gn_result * scale;

                // Max pool reduction
                if (scaled > max_val) {
                    max_val = scaled;
                    valid = true;
                }
            }
        }

        // Clamp
        if (valid) {
            max_val = fmaxf(clamp_min, fminf(clamp_max, max_val));
            int out_idx = batch_idx * (Co * mp_Ho * mp_Wo) + out_ch * (mp_Ho * mp_Wo) + out_row * mp_Wo + out_col;
            output[out_idx] = max_val;
        }
    }
}

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    torch::Tensor output,
    int gn_groups,
    float gn_eps,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int mp_kh, int mp_kw,
    int mp_sh, int mp_sw,
    int mp_ph, int mp_pw,
    float scale,
    float clamp_min,
    float clamp_max
) {
    int B = input.size(0);
    int Ci = input.size(1);
    int Hi = input.size(2);
    int Wi = input.size(3);
    int Co = conv_weight.size(0);
    int Kh = conv_weight.size(2);
    int Kw = conv_weight.size(3);

    dim3 grid((Hi + 2 * pad_h - Kh) / stride_h + 1, Co, B);
    dim3 block(256);

    int shared_mem_size = Co * ((Hi + 2 * pad_h - Kh) / stride_h + 1) * ((Wi + 2 * pad_w - Kw) / stride_w + 1) * sizeof(float);

    fused_op_forward_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, Ci, Hi, Wi,
        Co, Kh, Kw,
        stride_h, stride_w,
        pad_h, pad_w,
        gn_groups, gn_eps,
        mp_kh, mp_kw,
        mp_sh, mp_sw,
        mp_ph, mp_pw,
        scale,
        clamp_min,
        clamp_max
    );
}
"""

# C++ interface bindings
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    torch::Tensor output,
    int gn_groups,
    float gn_eps,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int mp_kh, int mp_kw,
    int mp_sh, int mp_sw,
    int mp_ph, int mp_pw,
    float scale,
    float clamp_min,
    float clamp_max
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv + GroupNorm + Scale + MaxPool + Clamp");
}
"""

# Compile extension
fused_ext = load_inline(
    name='fused_op_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# Optimized functional model using fused kernel
def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
    group_norm_weight,
    group_norm_bias,
    group_norm_num_groups,
    group_norm_eps,
    maxpool_kernel_size,
    maxpool_stride,
    maxpool_padding,
    maxpool_dilation,
    maxpool_ceil_mode,
    maxpool_return_indices,
    scale,
    clamp_min,
    clamp_max,
):
    # Ensure tensors are on CUDA
    x = x.cuda()
    conv_weight = conv_weight.cuda()
    conv_bias = conv_bias.cuda()
    group_norm_weight = group_norm_weight.cuda()
    group_norm_bias = group_norm_bias.cuda()

    # Calculate output dimensions after conv and maxpool
    batch_size = x.size(0)
    in_height, in_width = x.size(2), x.size(3)
    out_channels = conv_weight.size(0)
    conv_kh, conv_kw = conv_weight.size(2), conv_weight.size(3)

    # Conv output size
    conv_out_h = (in_height + 2 * conv_padding[0] - conv_dilation[0] * (conv_kh - 1) - 1) // conv_stride[0] + 1
    conv_out_w = (in_width + 2 * conv_padding[1] - conv_dilation[1] * (conv_kw - 1) - 1) // conv_stride[1] + 1

    # MaxPool output size
    maxpool_kh = maxpool_kw = maxpool_kernel_size
    maxpool_sh = maxpool_sw = maxpool_stride
    maxpool_ph = maxpool_pw = maxpool_padding[0]  # Assuming symmetric padding

    out_h = (conv_out_h + 2 * maxpool_ph - maxpool_dilation[0] * (maxpool_kh - 1) - 1) // maxpool_sh + 1
    out_w = (conv_out_w + 2 * maxpool_pw - maxpool_dilation[1] * (maxpool_kw - 1) - 1) // maxpool_sw + 1

    output = torch.empty((batch_size, out_channels, out_h, out_w), device='cuda')

    # Launch fused kernel
    fused_ext.fused_op(
        x, conv_weight, conv_bias,
        group_norm_weight, group_norm_bias,
        output,
        group_norm_num_groups, group_norm_eps,
        conv_stride[0], conv_stride[1],
        conv_padding[0], conv_padding[1],
        maxpool_kh, maxpool_kw,
        maxpool_sh, maxpool_sw,
        maxpool_ph, maxpool_pw,
        float(scale),
        float(clamp_min),
        float(clamp_max)
    )

    return output

# Constants (as provided in original)
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128 
kernel_size = 3
num_groups = 16
scale_shape = (out_channels, 1, 1)
maxpool_kernel_size = 4
clamp_min = 0.0
clamp_max = 1.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
