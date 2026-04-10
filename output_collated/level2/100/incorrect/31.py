# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_130522/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'min_value', 'divisor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'min_value', 'divisor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

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
    if 'min_value' in flat_state:
        state_kwargs['min_value'] = flat_state['min_value']
    else:
        state_kwargs['min_value'] = getattr(model, 'min_value')
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------------------------
# Inline CUDA kernel for fused 3D transposed convolution + clamp + division
# -------------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Element-wise activation and division kernel
__global__ void fused_activation_div_kernel(float* data, float min_val, float rdiv, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = data[idx];
        val = fmaxf(val, min_val); // Clamp to minimum (branchless)
        val = val * rdiv;          // Multiply by reciprocal instead of dividing
        data[idx] = val;
    }
}

// Host wrapper for activation/division fusion
void launch_fused_activation_div(torch::Tensor data, float min_val, float divisor) {
    float rdiv = 1.0f / divisor;
    int N = static_cast<int>(data.numel());
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;

    fused_activation_div_kernel<<<grid_size, block_size>>>(
        data.data_ptr<float>(), min_val, rdiv, N);
    cudaDeviceSynchronize();
}

// Helper to compute output dimensions for transposed convolution
__host__ __device__ inline int deconv_out_dim(int input_dim, int kernel_size, int stride, int padding, int output_padding, int dilation) {
    return (input_dim - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
}

// Custom 3D transposed convolution implementation
__global__ void conv_transpose3d_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch_size,
    int in_ch, int out_ch,
    int in_d, int in_h, int in_w,
    int k_d, int k_h, int k_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int out_d, int out_h, int out_w
) {
    int out_ch_id = blockIdx.x;
    int batch_id = blockIdx.y;
    int out_pos = blockIdx.z * blockDim.x + threadIdx.x;

    if (out_ch_id >= out_ch || batch_id >= batch_size || out_pos >= out_d * out_h * out_w)
        return;

    int od = out_pos / (out_h * out_w);
    int oh = (out_pos / out_w) % out_h;
    int ow = out_pos % out_w;

    float sum = 0.0f;

    for (int ic = 0; ic < in_ch; ++ic) {
        for (int kd = 0; kd < k_d; ++kd) {
            for (int kh = 0; kh < k_h; ++kh) {
                for (int kw = 0; kw < k_w; ++kw) {
                    int id = od + pad_d - kd * dilation_d;
                    int ih = oh + pad_h - kh * dilation_h;
                    int iw = ow + pad_w - kw * dilation_w;

                    if (id % stride_d == 0 && ih % stride_h == 0 && iw % stride_w == 0) {
                        id /= stride_d;
                        ih /= stride_h;
                        iw /= stride_w;

                        if (id >= 0 && id < in_d && ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            int input_idx = batch_id * (in_ch * in_d * in_h * in_w) +
                                            ic * (in_d * in_h * in_w) +
                                            id * (in_h * in_w) +
                                            ih * in_w +
                                            iw;

                            int weight_idx = out_ch_id * (in_ch * k_d * k_h * k_w) +
                                             ic * (k_d * k_h * k_w) +
                                             kd * (k_h * k_w) +
                                             kh * k_w +
                                             kw;

                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }

    int output_idx = batch_id * (out_ch * out_d * out_h * out_w) +
                     out_ch_id * (out_d * out_h * out_w) +
                     od * (out_h * out_w) +
                     oh * out_w +
                     ow;

    output[output_idx] = sum + bias[out_ch_id];
}
"""

# -------------------------------------------------------------------------
# C++ binding for the CUDA kernels
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void launch_fused_activation_div(torch::Tensor data, float min_val, float divisor);
void conv_transpose3d_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch_size,
    int in_ch, int out_ch,
    int in_d, int in_h, int in_w,
    int k_d, int k_h, int k_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int out_d, int out_h, int out_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_activation_div", &launch_fused_activation_div, "Fused clamp(min) + division");
    m.def("conv_transpose3d_custom", &conv_transpose3d_kernel, "Custom 3D transposed convolution kernel");
}
"""

# -------------------------------------------------------------------------
# Build the inline extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='optimized_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Problem size constants
# -------------------------------------------------------------------------
batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 24, 48, 48
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0

def get_init_inputs():
    """Used by the test harness to obtain the fixed parameters."""
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]

def get_inputs():
    """Creates a random input tensor (batch, in_channels, depth, height, width)."""
    return [torch.rand(batch_size, in_channels, depth, height, width)]

# -------------------------------------------------------------------------
# The optimized functional_model using custom CUDA kernels
# -------------------------------------------------------------------------
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
    min_value,
    divisor,
):
    """
    Performs a custom 3D transposed convolution followed by fused clamp(min) + division.
    """
    # Ensure tensors are on GPU
    if not x.is_cuda:
        x = x.cuda()
    if not conv_transpose_weight.is_cuda:
        conv_transpose_weight = conv_transpose_weight.cuda()
    if not conv_transpose_bias.is_cuda:
        conv_transpose_bias = conv_transpose_bias.cuda()

    # Calculate output dimensions
    out_d = (depth - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + \
            conv_transpose_dilation[0] * (kernel_size - 1) + conv_transpose_output_padding[0] + 1
    out_h = (height - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + \
            conv_transpose_dilation[1] * (kernel_size - 1) + conv_transpose_output_padding[1] + 1
    out_w = (width - 1) * conv_transpose_stride[2] - 2 * conv_transpose_padding[2] + \
            conv_transpose_dilation[2] * (kernel_size - 1) + conv_transpose_output_padding[2] + 1

    # Create output tensor
    output_shape = (batch_size, out_channels, out_d, out_h, out_w)
    output = torch.empty(output_shape, device=x.device, dtype=x.dtype)

    # Launch custom 3D transposed convolution kernel
    grid = (out_channels, batch_size, (out_d * out_h * out_w + 255) // 256)
    block = (256, 1, 1)

    fused_ext.conv_transpose3d_custom(
        x.data_ptr<float>(),
        conv_transpose_weight.data_ptr<float>(),
        conv_transpose_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        depth, height, width,
        kernel_size, kernel_size, kernel_size,
        conv_transpose_stride[0], conv_transpose_stride[1], conv_transpose_stride[2],
        conv_transpose_padding[0], conv_transpose_padding[1], conv_transpose_padding[2],
        conv_transpose_output_padding[0], conv_transpose_output_padding[1], conv_transpose_output_padding[2],
        conv_transpose_dilation[0], conv_transpose_dilation[1], conv_transpose_dilation[2],
        out_d, out_h, out_w
    )

    # Launch fused activation/division kernel
    fused_ext.fused_activation_div(output, min_value, divisor)

    return output
