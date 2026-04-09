# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_034836/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'max_pool1_kernel_size', 'max_pool1_stride', 'max_pool1_padding', 'max_pool1_dilation', 'max_pool1_ceil_mode', 'max_pool1_return_indices', 'max_pool2_kernel_size', 'max_pool2_stride', 'max_pool2_padding', 'max_pool2_dilation', 'max_pool2_ceil_mode', 'max_pool2_return_indices']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)

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
    # State for max_pool1 (nn.MaxPool3d)
    state_kwargs['max_pool1_kernel_size'] = model.max_pool1.kernel_size
    state_kwargs['max_pool1_stride'] = model.max_pool1.stride
    state_kwargs['max_pool1_padding'] = model.max_pool1.padding
    state_kwargs['max_pool1_dilation'] = model.max_pool1.dilation
    state_kwargs['max_pool1_ceil_mode'] = model.max_pool1.ceil_mode
    state_kwargs['max_pool1_return_indices'] = model.max_pool1.return_indices
    # State for max_pool2 (nn.MaxPool3d)
    state_kwargs['max_pool2_kernel_size'] = model.max_pool2.kernel_size
    state_kwargs['max_pool2_stride'] = model.max_pool2.stride
    state_kwargs['max_pool2_padding'] = model.max_pool2.padding
    state_kwargs['max_pool2_dilation'] = model.max_pool2.dilation
    state_kwargs['max_pool2_ceil_mode'] = model.max_pool2.ceil_mode
    state_kwargs['max_pool2_return_indices'] = model.max_pool2.return_indices
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
# CUDA kernel source (fused conv_transpose3d + 2 max_pool3d + sum)
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// ------------------------------------------------------------
// Fused kernel: conv_transpose3d -> max_pool3d -> max_pool3d -> sum
// ------------------------------------------------------------

__global__ void fused_conv_transpose_pool_pool_sum_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out,
    int iD, int iH, int iW,
    int kernel_size, int conv_stride, int conv_padding, int conv_output_padding, int conv_dilation,
    int pool1_kernel_size, int pool1_stride, int pool1_padding, int pool1_dilation, bool pool1_ceil_mode,
    int pool2_kernel_size, int pool2_stride, int pool2_padding, int pool2_dilation, bool pool2_ceil_mode)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate output dimensions
    int conv_oD = (iD - 1) * conv_stride - 2 * conv_padding + conv_dilation * (kernel_size - 1) + conv_output_padding + 1;
    int conv_oH = (iH - 1) * conv_stride - 2 * conv_padding + conv_dilation * (kernel_size - 1) + conv_output_padding + 1;
    int conv_oW = (iW - 1) * conv_stride - 2 * conv_padding + conv_dilation * (kernel_size - 1) + conv_output_padding + 1;
    
    int pool1_oD, pool1_oH, pool1_oW;
    if (pool1_ceil_mode) {
        pool1_oD = (conv_oD + 2 * pool1_padding - pool1_dilation * (pool1_kernel_size - 1) - 1 + pool1_stride - 1) / pool1_stride + 1;
        pool1_oH = (conv_oH + 2 * pool1_padding - pool1_dilation * (pool1_kernel_size - 1) - 1 + pool1_stride - 1) / pool1_stride + 1;
        pool1_oW = (conv_oW + 2 * pool1_padding - pool1_dilation * (pool1_kernel_size - 1) - 1 + pool1_stride - 1) / pool1_stride + 1;
    } else {
        pool1_oD = (conv_oD + 2 * pool1_padding - pool1_dilation * (pool1_kernel_size - 1) - 1) / pool1_stride + 1;
        pool1_oH = (conv_oH + 2 * pool1_padding - pool1_dilation * (pool1_kernel_size - 1) - 1) / pool1_stride + 1;
        pool1_oW = (conv_oW + 2 * pool1_padding - pool1_dilation * (pool1_kernel_size - 1) - 1) / pool1_stride + 1;
    }
    
    int pool2_oD, pool2_oH, pool2_oW;
    if (pool2_ceil_mode) {
        pool2_oD = (pool1_oD + 2 * pool2_padding - pool2_dilation * (pool2_kernel_size - 1) - 1 + pool2_stride - 1) / pool2_stride + 1;
        pool2_oH = (pool1_oH + 2 * pool2_padding - pool2_dilation * (pool2_kernel_size - 1) - 1 + pool2_stride - 1) / pool2_stride + 1;
        pool2_oW = (pool1_oW + 2 * pool2_padding - pool2_dilation * (pool2_kernel_size - 1) - 1 + pool2_stride - 1) / pool2_stride + 1;
    } else {
        pool2_oD = (pool1_oD + 2 * pool2_padding - pool2_dilation * (pool2_kernel_size - 1) - 1) / pool2_stride + 1;
        pool2_oH = (pool1_oH + 2 * pool2_padding - pool2_dilation * (pool2_kernel_size - 1) - 1) / pool2_stride + 1;
        pool2_oW = (pool1_oW + 2 * pool2_padding - pool2_dilation * (pool2_kernel_size - 1) - 1) / pool2_stride + 1;
    }

    int total_output_elements = N * pool2_oD * pool2_oH * pool2_oW;
    if (tid >= total_output_elements) return;

    // Decode output coordinates
    int n = tid / (pool2_oD * pool2_oH * pool2_oW);
    int rem = tid % (pool2_oD * pool2_oH * pool2_oW);
    int out_d = rem / (pool2_oH * pool2_oW);
    int out_h = (rem % (pool2_oH * pool2_oW)) / pool2_oW;
    int out_w = rem % pool2_oW;

    // Backward map to pool1 coordinates
    int start_d2 = out_d * pool2_stride - pool2_padding;
    int start_h2 = out_h * pool2_stride - pool2_padding;
    int start_w2 = out_w * pool2_stride - pool2_padding;

    float final_sum = 0.0f;

    // Loop over pool2 kernel
    for (int kd2 = 0; kd2 < pool2_kernel_size; ++kd2) {
        int pd = start_d2 + kd2 * pool2_dilation;
        if (pd < 0 || pd >= pool1_oD) continue;
        for (int kh2 = 0; kh2 < pool2_kernel_size; ++kh2) {
            int ph = start_h2 + kh2 * pool2_dilation;
            if (ph < 0 || ph >= pool1_oH) continue;
            for (int kw2 = 0; kw2 < pool2_kernel_size; ++kw2) {
                int pw = start_w2 + kw2 * pool2_dilation;
                if (pw < 0 || pw >= pool1_oW) continue;

                // Backward map to conv coordinates
                int start_d1 = pd * pool1_stride - pool1_padding;
                int start_h1 = ph * pool1_stride - pool1_padding;
                int start_w1 = pw * pool1_stride - pool1_padding;

                float max_val1 = -FLT_MAX;

                // Loop over pool1 kernel
                for (int kd1 = 0; kd1 < pool1_kernel_size; ++kd1) {
                    int cd = start_d1 + kd1 * pool1_dilation;
                    if (cd < 0 || cd >= conv_oD) continue;
                    for (int kh1 = 0; kh1 < pool1_kernel_size; ++kh1) {
                        int ch = start_h1 + kh1 * pool1_dilation;
                        if (ch < 0 || ch >= conv_oH) continue;
                        for (int kw1 = 0; kw1 < pool1_kernel_size; ++kw1) {
                            int cw = start_w1 + kw1 * pool1_dilation;
                            if (cw < 0 || cw >= conv_oW) continue;

                            // Backward map to input coordinates for convolution
                            int iD_start = cd * conv_stride - conv_padding;
                            int iH_start = ch * conv_stride - conv_padding;
                            int iW_start = cw * conv_stride - conv_padding;

                            float conv_sum = 0.0f;
                            if (bias) {
                                for (int c_out = 0; c_out < C_out; ++c_out) {
                                    conv_sum += bias[c_out];
                                }
                            }

                            // Convolution loop
                            for (int kd = 0; kd < kernel_size; ++kd) {
                                int id = iD_start + kd * conv_dilation;
                                if (id >= 0 && id < iD) {
                                    for (int kh = 0; kh < kernel_size; ++kh) {
                                        int ih = iH_start + kh * conv_dilation;
                                        if (ih >= 0 && ih < iH) {
                                            for (int kw = 0; kw < kernel_size; ++kw) {
                                                int iw = iW_start + kw * conv_dilation;
                                                if (iw >= 0 && iw < iW) {
                                                    // Inner channel loop
                                                    for (int c_in = 0; c_in < C_in; ++c_in) {
                                                        for (int c_out = 0; c_out < C_out; ++c_out) {
                                                            int weight_idx = ((c_out * C_in + c_in) * kernel_size + kd) *
                                                                             (kernel_size * kernel_size) + kh * kernel_size + kw;
                                                            int input_idx = ((n * C_in + c_in) * iD + id) *
                                                                            (iH * iW) + ih * iW + iw;
                                                            conv_sum += weight[weight_idx] * input[input_idx];
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            if (conv_sum > max_val1) max_val1 = conv_sum;
                        }
                    }
                }
                final_sum += max_val1;
            }
        }
    }
    output[tid] = final_sum;
}

// ------------------------------------------------------------
// C++ wrapper that launches the fused kernel
// ------------------------------------------------------------
void fused_conv_transpose_pool_pool_sum_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int conv_stride, int conv_padding, int conv_output_padding, int conv_dilation,
    int pool1_kernel_size, int pool1_stride, int pool1_padding, int pool1_dilation, bool pool1_ceil_mode,
    int pool2_kernel_size, int pool2_stride, int pool2_padding, int pool2_dilation, bool pool2_ceil_mode)
{
    int N = input.size(0);
    int C_in = input.size(1);
    int iD = input.size(2), iH = input.size(3), iW = input.size(4);
    int C_out = weight.size(0);
    int kernel_size = weight.size(2);

    // Calculate output dimensions
    int conv_oD = (iD - 1) * conv_stride - 2 * conv_padding + conv_dilation * (kernel_size - 1) + conv_output_padding + 1;
    int conv_oH = (iH - 1) * conv_stride - 2 * conv_padding + conv_dilation * (kernel_size - 1) + conv_output_padding + 1;
    int conv_oW = (iW - 1) * conv_stride - 2 * conv_padding + conv_dilation * (kernel_size - 1) + conv_output_padding + 1;
    
    int pool1_oD, pool1_oH, pool1_oW;
    if (pool1_ceil_mode) {
        pool1_oD = (conv_oD + 2 * pool1_padding - pool1_dilation * (pool1_kernel_size - 1) - 1 + pool1_stride - 1) / pool1_stride + 1;
        pool1_oH = (conv_oH + 2 * pool1_padding - pool1_dilation * (pool1_kernel_size - 1) - 1 + pool1_stride - 1) / pool1_stride + 1;
        pool1_oW = (conv_oW + 2 * pool1_padding - pool1_dilation * (pool1_kernel_size - 1) - 1 + pool1_stride - 1) / pool1_stride + 1;
    } else {
        pool1_oD = (conv_oD + 2 * pool1_padding - pool1_dilation * (pool1_kernel_size - 1) - 1) / pool1_stride + 1;
        pool1_oH = (conv_oH + 2 * pool1_padding - pool1_dilation * (pool1_kernel_size - 1) - 1) / pool1_stride + 1;
        pool1_oW = (conv_oW + 2 * pool1_padding - pool1_dilation * (pool1_kernel_size - 1) - 1) / pool1_stride + 1;
    }
    
    int pool2_oD, pool2_oH, pool2_oW;
    if (pool2_ceil_mode) {
        pool2_oD = (pool1_oD + 2 * pool2_padding - pool2_dilation * (pool2_kernel_size - 1) - 1 + pool2_stride - 1) / pool2_stride + 1;
        pool2_oH = (pool1_oH + 2 * pool2_padding - pool2_dilation * (pool2_kernel_size - 1) - 1 + pool2_stride - 1) / pool2_stride + 1;
        pool2_oW = (pool1_oW + 2 * pool2_padding - pool2_dilation * (pool2_kernel_size - 1) - 1 + pool2_stride - 1) / pool2_stride + 1;
    } else {
        pool2_oD = (pool1_oD + 2 * pool2_padding - pool2_dilation * (pool2_kernel_size - 1) - 1) / pool2_stride + 1;
        pool2_oH = (pool1_oH + 2 * pool2_padding - pool2_dilation * (pool2_kernel_size - 1) - 1) / pool2_stride + 1;
        pool2_oW = (pool1_oW + 2 * pool2_padding - pool2_dilation * (pool2_kernel_size - 1) - 1) / pool2_stride + 1;
    }

    const float* in_d    = input.data_ptr<float>();
    const float* w_d     = weight.data_ptr<float>();
    const float* b_d     = bias.numel() ? bias.data_ptr<float>() : nullptr;
    float* out_d         = output.data_ptr<float>();

    int total = N * pool2_oD * pool2_oH * pool2_oW;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fused_conv_transpose_pool_pool_sum_kernel<<<blocks, threads>>>(
        in_d, w_d, b_d, out_d,
        N, C_in, C_out, iD, iH, iW,
        kernel_size, conv_stride, conv_padding, conv_output_padding, conv_dilation,
        pool1_kernel_size, pool1_stride, pool1_padding, pool1_dilation, pool1_ceil_mode,
        pool2_kernel_size, pool2_stride, pool2_padding, pool2_dilation, pool2_ceil_mode);
    cudaDeviceSynchronize();
}
"""

# ----------------------------------------------------------------------
# C++ binding (PYBIND11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_pool_pool_sum_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int conv_stride, int conv_padding, int conv_output_padding, int conv_dilation,
    int pool1_kernel_size, int pool1_stride, int pool1_padding, int pool1_dilation, bool pool1_ceil_mode,
    int pool2_kernel_size, int pool2_stride, int pool2_padding, int pool2_dilation, bool pool2_ceil_mode);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_pool_pool_sum_forward", &fused_conv_transpose_pool_pool_sum_forward, "Fused conv transpose + pool + pool + sum forward");
}
"""

# ----------------------------------------------------------------------
# Compile the custom CUDA extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Helper functions for computing output sizes
# ----------------------------------------------------------------------
def pool_out_size(in_size, kernel_size, stride, padding, dilation, ceil_mode=False):
    if ceil_mode:
        return (in_size + 2 * padding - dilation * (kernel_size - 1) - 1 + stride - 1) // stride + 1
    else:
        return (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

def conv_transpose_out_size(in_size, stride, padding, kernel_size, dilation, output_padding):
    return (in_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

# ----------------------------------------------------------------------
# The optimized functional_model
# ----------------------------------------------------------------------
def functional_model(
    x,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    max_pool1_kernel_size,
    max_pool1_stride,
    max_pool1_padding,
    max_pool1_dilation,
    max_pool1_ceil_mode,
    max_pool1_return_indices,
    max_pool2_kernel_size,
    max_pool2_stride,
    max_pool2_padding,
    max_pool2_dilation,
    max_pool2_ceil_mode,
    max_pool2_return_indices,
):
    # ------------------------------------------------------------
    # 1) Move all data to the GPU
    # ------------------------------------------------------------
    device = torch.device('cuda')
    x = x.to(device).contiguous()
    w = conv_transpose_weight.to(device).contiguous()
    if conv_transpose_bias is not None:
        b = conv_transpose_bias.to(device).contiguous()
    else:
        b = torch.zeros(w.size(0), dtype=torch.float32, device=device)

    # ------------------------------------------------------------
    # 2) Calculate output dimensions
    # ------------------------------------------------------------
    in_d, in_h, in_w = x.size(2), x.size(3), x.size(4)
    kernel_size = w.size(2)  # square kernel assumed
    
    out_d = conv_transpose_out_size(in_d, conv_transpose_stride,
                                    conv_transpose_padding, kernel_size,
                                    conv_transpose_dilation,
                                    conv_transpose_output_padding)
    out_h = conv_transpose_out_size(in_h, conv_transpose_stride,
                                    conv_transpose_padding, kernel_size,
                                    conv_transpose_dilation,
                                    conv_transpose_output_padding)
    out_w = conv_transpose_out_size(in_w, conv_transpose_stride,
                                    conv_transpose_padding, kernel_size,
                                    conv_transpose_dilation,
                                    conv_transpose_output_padding)

    out_d1 = pool_out_size(out_d, max_pool1_kernel_size, max_pool1_stride,
                           max_pool1_padding, max_pool1_dilation, max_pool1_ceil_mode)
    out_h1 = pool_out_size(out_h, max_pool1_kernel_size, max_pool1_stride,
                           max_pool1_padding, max_pool1_dilation, max_pool1_ceil_mode)
    out_w1 = pool_out_size(out_w, max_pool1_kernel_size, max_pool1_stride,
                           max_pool1_padding, max_pool1_dilation, max_pool1_ceil_mode)

    out_d2 = pool_out_size(out_d1, max_pool2_kernel_size, max_pool2_stride,
                           max_pool2_padding, max_pool2_dilation, max_pool2_ceil_mode)
    out_h2 = pool_out_size(out_h1, max_pool2_kernel_size, max_pool2_stride,
                           max_pool2_padding, max_pool2_dilation, max_pool2_ceil_mode)
    out_w2 = pool_out_size(out_w1, max_pool2_kernel_size, max_pool2_stride,
                           max_pool2_padding, max_pool2_dilation, max_pool2_ceil_mode)

    # ------------------------------------------------------------
    # 3) Allocate output tensor
    # ------------------------------------------------------------
    final = torch.empty(x.size(0), 1, out_d2, out_h2, out_w2,
                        device=device, dtype=x.dtype)

    # ------------------------------------------------------------
    # 4) Launch fused kernel
    # ------------------------------------------------------------
    fused_ext.fused_conv_transpose_pool_pool_sum_forward(
        x, w, b, final,
        conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding, conv_transpose_dilation,
        max_pool1_kernel_size, max_pool1_stride, max_pool1_padding, max_pool1_dilation, max_pool1_ceil_mode,
        max_pool2_kernel_size, max_pool2_stride, max_pool2_padding, max_pool2_dilation, max_pool2_ceil_mode)

    return final

batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 32, 32, 32
kernel_size = 5
stride = 2
padding = 2

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
