# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_123708/code_3.py
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
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# CUDA source – fused transposed‑convolution + clamp + scale
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_transpose_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int K, const int stride, const int padding,
    const int dilation, const int output_padding,
    const float min_val, const float divisor)
{
    const int total_out = N * C_out * D_out * H_out * W_out;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride_grid = blockDim.x * gridDim.x;

    // Grid‑stride loop – each thread may produce several output points
    for (int idx = tid; idx < total_out; idx += stride_grid) {
        // Unpack linear index to (n, co, d, h, w)
        int tmp = idx;
        int n = tmp / (C_out * D_out * H_out * W_out);
        tmp %= (C_out * D_out * H_out * W_out);
        int co = tmp / (D_out * H_out * W_out);
        tmp %= (D_out * H_out * W_out);
        int d = tmp / (H_out * W_out);
        tmp %= (H_out * W_out);
        int h = tmp / W_out;
        int w = tmp % W_out;

        // Start accumulator with bias
        float acc = bias[co];

        // Loop over input channels and kernel positions
        // The kernel size is 3 – we unroll the three innermost loops
        for (int ci = 0; ci < C_in; ++ci) {
            // Base pointer for this (co,ci) block in the weight tensor
            const int wbase = ((co * C_in + ci) * K * K * K);

            #pragma unroll
            for (int kd = 0; kd < K; ++kd) {
                int d_in = d + padding - kd * dilation;
                if (d_in < 0 || d_in % stride != 0) continue;
                d_in /= stride;
                if (d_in >= D_in) continue;

                #pragma unroll
                for (int kh = 0; kh < K; ++kh) {
                    int h_in = h + padding - kh * dilation;
                    if (h_in < 0 || h_in % stride != 0) continue;
                    h_in /= stride;
                    if (h_in >= H_in) continue;

                    #pragma unroll
                    for (int kw = 0; kw < K; ++kw) {
                        int w_in = w + padding - kw * dilation;
                        if (w_in < 0 || w_in % stride != 0) continue;
                        w_in /= stride;
                        if (w_in >= W_in) continue;

                        // weight index
                        int wIdx = wbase + ((kd * K + kh) * K + kw);
                        float wval = weight[wIdx];

                        // input index (NCHWD layout)
                        int iIdx = (((n * C_in + ci) * D_in + d_in) * H_in + h_in) * W_in + w_in;
                        float ival = input[iIdx];

                        acc += ival * wval;
                    }
                }
            }
        }

        // Clamp and scale
        if (acc < min_val) acc = min_val;
        acc /= divisor;

        // Write output (coalesced)
        int oIdx = (((n * C_out + co) * D_out + d) * H_out + h) * W_out + w;
        output[oIdx] = acc;
    }
}

// Wrapper that chooses a reasonable grid size
void fused_transpose_conv(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int N, int C_in, int C_out,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int K, int stride, int padding,
    int dilation, int output_padding,
    float min_val, float divisor)
{
    const int total_out = N * C_out * D_out * H_out * W_out;
    const int threads = 256;
    const int max_blocks = 65535;
    const int blocks = (total_out + threads - 1) / threads;
    const int grid_x = (blocks > max_blocks) ? max_blocks : blocks;
    const int grid_y = (blocks + grid_x - 1) / grid_x;

    fused_transpose_conv_kernel<<<dim3(grid_x, grid_y, 1), threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        K, stride, padding,
        dilation, output_padding,
        min_val, divisor);

    cudaDeviceSynchronize();
}
"""

# ----------------------------------------------------------------------
# C++ binding (PYBIND11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_transpose_conv(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int N, int C_in, int C_out,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int K, int stride, int padding,
    int dilation, int output_padding,
    float min_val, float divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_transpose_conv", &fused_transpose_conv,
          "Fused transposed convolution + clamp + scale");
}
"""

# ----------------------------------------------------------------------
# Compile the extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_transpose_conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# ----------------------------------------------------------------------
# Model parameters (identical to the original script)
# ----------------------------------------------------------------------
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
    return [in_channels, out_channels, kernel_size, stride, padding,
            min_value, divisor]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

# ----------------------------------------------------------------------
# Functional model – now uses the fused CUDA kernel
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
    min_value,
    divisor,
):
    # Move data to GPU if not already there
    if not x.is_cuda:
        x = x.cuda()
    if not conv_transpose_weight.is_cuda:
        conv_transpose_weight = conv_transpose_weight.cuda()
    if not conv_transpose_bias.is_cuda:
        conv_transpose_bias = conv_transpose_bias.cuda()

    N, C_in, D_in, H_in, W_in = x.shape
    # Compute output shape using PyTorch's formula
    K = conv_transpose_weight.shape[2]          # kernel size (3)
    C_out = conv_transpose_weight.shape[0]      # out_channels
    D_out = (D_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + \
            conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1
    H_out = (H_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + \
            conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1
    W_out = (W_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + \
            conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1

    # Allocate output tensor
    output = torch.empty((N, C_out, D_out, H_out, W_out),
                        dtype=x.dtype, device=x.device)

    # Launch fused kernel
    fused_ext.fused_transpose_conv(
        x, conv_transpose_weight, conv_transpose_bias, output,
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        K, conv_transpose_stride, conv_transpose_padding,
        conv_transpose_dilation, conv_transpose_output_padding,
        min_value, divisor)

    return output
