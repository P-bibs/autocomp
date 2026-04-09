# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_114641/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'bias_shape', 'stride', 'padding', 'output_padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

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
# 1.  CUDA source – fused transposed‑conv + bias + tanh kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_transpose_conv_bias_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ final_bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int H_in, const int W_in,
    const int H_out, const int W_out,
    const int stride, const int padding,
    const int dilation, const int kernel_size,
    const int /*output_padding – already accounted in H_out/W_out*/)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    // ---- decode flat index to (n, co, ho, wo) ----
    int rem = idx;
    const int n = rem / (C_out * H_out * W_out);
    rem = rem % (C_out * H_out * W_out);
    const int co = rem / (H_out * W_out);
    rem = rem % (H_out * W_out);
    const int ho = rem / W_out;
    const int wo = rem % W_out;

    float sum = 0.0f;

    // ---- transposed convolution loops ----
    for (int ci = 0; ci < C_in; ++ci) {
        const int weight_base = ((ci * C_out + co) * kernel_size) * kernel_size;
        for (int kh = 0; kh < kernel_size; ++kh) {
            int ho_shifted = ho + padding - kh * dilation;
            if (ho_shifted < 0) continue;
            if (ho_shifted % stride != 0) continue;
            int hi = ho_shifted / stride;
            if (hi >= H_in) continue;

            for (int kw = 0; kw < kernel_size; ++kw) {
                int wo_shifted = wo + padding - kw * dilation;
                if (wo_shifted < 0) continue;
                if (wo_shifted % stride != 0) continue;
                int wi = wo_shifted / stride;
                if (wi >= W_in) continue;

                // ---- load input (use read‑only cache) ----
                const int in_idx = ((n * C_in + ci) * H_in + hi) * W_in + wi;
                float inp = __ldg(&input[in_idx]);

                // ---- load weight (use read‑only cache) ----
                const int w_idx = weight_base + kh * kernel_size + kw;
                float w = __ldg(&weight[w_idx]);

                sum += inp * w;
            }
        }
    }

    // ---- conv bias (added inside the convolution) ----
    sum += __ldg(&conv_bias[co]);

    // ---- extra bias subtraction ----
    sum -= __ldg(&final_bias[co]);

    // ---- tanh activation (fast math) ----
    sum = tanhf(sum);

    // ---- write result ----
    const int out_idx = ((n * C_out + co) * H_out + ho) * W_out + wo;
    output[out_idx] = sum;
}

// ---- host wrapper that launches the kernel ----
void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor final_bias,
    torch::Tensor output,
    int N, int C_in, int C_out,
    int H_in, int W_in,
    int H_out, int W_out,
    int stride, int padding,
    int dilation, int kernel_size,
    int output_padding)
{
    const int total = N * C_out * H_out * W_out;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    fused_transpose_conv_bias_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        final_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out,
        H_in, W_in,
        H_out, W_out,
        stride, padding,
        dilation, kernel_size,
        output_padding);
    cudaDeviceSynchronize();
}
"""

# ----------------------------------------------------------------------
# 2.  C++ binding – exposes the wrapper to Python
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor final_bias,
    torch::Tensor output,
    int N, int C_in, int C_out,
    int H_in, int W_in,
    int H_out, int W_out,
    int stride, int padding,
    int dilation, int kernel_size,
    int output_padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward,
          "Fused transposed convolution + bias + tanh");
}
"""

# ----------------------------------------------------------------------
# 3.  Build the extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_transpose_conv',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# 4.  Helper to compute output spatial size (same formula as PyTorch)
# ----------------------------------------------------------------------
def conv_transpose_output_size(x, stride, padding, output_padding,
                               dilation, kernel_size):
    H = x.size(2)
    W = x.size(3)
    H_out = (H - 1) * stride - 2 * padding + \
            dilation * (kernel_size - 1) + output_padding + 1
    W_out = (W - 1) * stride - 2 * padding + \
            dilation * (kernel_size - 1) + output_padding + 1
    return H_out, W_out

# ----------------------------------------------------------------------
# 5.  The fused functional_model – replaces the three‑kernel version
# ----------------------------------------------------------------------
def functional_model(
    x,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,      # ignored – we assume groups==1
    conv_transpose_dilation,
    bias,                       # the extra bias that is subtracted
):
    # Ensure contiguous GPU tensors
    x = x.contiguous()
    weight = conv_transpose_weight.contiguous()
    conv_bias = conv_transpose_bias.contiguous()
    final_bias = bias.contiguous().view(-1)   # (out_channels,)

    # Spatial output size
    H_out, W_out = conv_transpose_output_size(
        x, conv_transpose_stride, conv_transpose_padding,
        conv_transpose_output_padding, conv_transpose_dilation,
        conv_transpose_weight.size(2)   # kernel height (=width)
    )

    N = x.size(0)
    C_in = x.size(1)
    C_out = conv_transpose_weight.size(1)

    # Allocate output
    output = torch.empty((N, C_out, H_out, W_out),
                        dtype=x.dtype, device=x.device)

    # Launch the fused kernel
    fused_ext.fused_op(
        x, weight, conv_bias, final_bias, output,
        N, C_in, C_out,
        x.size(2), x.size(3),
        H_out, W_out,
        conv_transpose_stride, conv_transpose_padding,
        conv_transpose_dilation,
        conv_transpose_weight.size(2),
        conv_transpose_output_padding
    )
    return output

# ----------------------------------------------------------------------
# 6.  Keep the original helper functions for the test harness
# ----------------------------------------------------------------------
batch_size = 32
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
