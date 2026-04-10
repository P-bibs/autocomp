# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_163844/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape', 'scaling_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias', 'scaling_factor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

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
    # State for conv_transpose (nn.ConvTranspose2d)
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
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
    if 'scaling_factor' in flat_state:
        state_kwargs['scaling_factor'] = flat_state['scaling_factor']
    else:
        state_kwargs['scaling_factor'] = getattr(model, 'scaling_factor')
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------------------------
# 1.  CUDA source – fused transposed convolution + element‑wise kernel
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>

// Element-wise operations fused kernel
__global__ void fused_post_op_kernel(
    float* __restrict__ input,
    const float* __restrict__ bias,
    float scaling_factor,
    int N, int C, int H, int W)
{
    // Grid-stride loop – one thread handles many elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = N * C * H * W;

    for (int i = idx; i < total; i += stride) {
        // Recover channel index from linear id (row-major order)
        int c = (i / (H * W)) % C;

        float val = input[i];

        // 1) add per-channel bias
        val += bias[c];

        // 2) first clamp to [0,1]
        val = fmaxf(0.0f, fminf(1.0f, val));

        // 3) multiply by scaling factor
        val *= scaling_factor;

        // 4) second clamp to [0,1]
        val = fmaxf(0.0f, fminf(1.0f, val));

        // 5) divide by scaling factor
        val /= scaling_factor;

        // Store back (in-place)
        input[i] = val;
    }
}

// Transposed convolution kernel implementation
__global__ void transposed_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int dilation
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * out_height * out_width;
    
    if (out_idx >= total_outputs) return;

    int w_out = out_idx % out_width;
    out_idx /= out_width;
    int h_out = out_idx % out_height;
    out_idx /= out_height;
    int c_out = out_idx % out_channels;
    int n = out_idx / out_channels;

    float sum = 0.0f;
    if (bias) {
        sum = bias[c_out];
    }

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h_out + padding - kh * dilation;
                int w_in = w_out + padding - kw * dilation;
                
                if ((h_in % stride == 0) && (w_in % stride == 0)) {
                    h_in /= stride;
                    w_in /= stride;
                    
                    if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                        int input_idx = ((n * in_channels + c_in) * in_height + h_in) * in_width + w_in;
                        int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    output[out_idx * out_height * out_width + h_out * out_width + w_out] = sum;
}

// Host-side wrapper called from Python
void fused_post_op(
    torch::Tensor input,
    torch::Tensor bias,
    float scaling_factor)
{
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    const int threads = 256;
    const int blocks = (N * C * H * W + threads - 1) / threads;

    fused_post_op_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        scaling_factor,
        N, C, H, W
    );
}

void transposed_conv2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int output_padding,
    int dilation
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;

    const int threads = 256;
    int total_outputs = batch_size * out_channels * out_height * out_width;
    const int blocks = (total_outputs + threads - 1) / threads;

    transposed_conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation
    );
}
"""

# -------------------------------------------------------------------------
# 2.  C++ interface (PyBind11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_post_op(
    torch::Tensor input,
    torch::Tensor bias,
    float scaling_factor);

void transposed_conv2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int output_padding,
    int dilation);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_op", &fused_post_op,
          "Fused bias-add + clamp*2 + scale element-wise kernel");
    m.def("transposed_conv2d", &transposed_conv2d,
          "Custom transposed convolution 2D kernel");
}
"""

# -------------------------------------------------------------------------
# 3.  Compile the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# 4.  functional_model – the only function imported for evaluation
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
    bias,
    scaling_factor,
):
    # ---- transposed convolution (custom CUDA kernel) --------------
    # Calculate output dimensions
    in_height, in_width = x.shape[2], x.shape[3]
    kernel_size = conv_transpose_weight.shape[2]
    
    out_height = (in_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + conv_transpose_output_padding + 1
    out_width = (in_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + conv_transpose_output_padding + 1
    
    out_channels = conv_transpose_weight.shape[0]
    batch_size = x.shape[0]
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Call custom transposed convolution
    fused_ext.transposed_conv2d(
        x,
        conv_transpose_weight,
        conv_transpose_bias if conv_transpose_bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        output,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_dilation
    )

    # ---- fuse the remaining element-wise ops into one kernel ----------
    # Ensure a contiguous layout for the custom kernel
    output = output.contiguous()
    # bias has shape (out_channels,1,1) → flatten to (out_channels,)
    bias_flat = bias.squeeze().contiguous()

    # Single fused kernel: bias add → clamp → scale → clamp → un-scale
    fused_ext.fused_post_op(output, bias_flat, scaling_factor)

    return output


# -------------------------------------------------------------------------
# 5.  Helper functions needed for the test harness
# -------------------------------------------------------------------------
batch_size = 128
in_channels = 64
out_channels = 64
height = width = 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape,
        scaling_factor,
    ]


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
