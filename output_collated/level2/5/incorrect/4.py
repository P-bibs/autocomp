# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_112403/code_0.py
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

# --- CUDA Kernel Definition ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>

template <typename scalar_t>
__global__ void fused_conv_transpose_tanh_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ conv_bias,
    const scalar_t* __restrict__ bias_tensor,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    int out_height,
    int out_width
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;

    if (tid >= total_elements) return;

    int n = tid / (out_channels * out_height * out_width);
    int c_out = (tid / (out_height * out_width)) % out_channels;
    int y_out = (tid / out_width) % out_height;
    int x_out = tid % out_width;

    scalar_t sum = 0.0;

    int kernel_h = kernel_size;
    int kernel_w = kernel_size;

    int group_id = c_out / (out_channels / groups);
    int group_start = group_id * (in_channels / groups);
    int channels_per_group = in_channels / groups;

    for (int c_in_idx = 0; c_in_idx < channels_per_group; ++c_in_idx) {
        int c_in = group_start + c_in_idx;

        for (int ky = 0; ky < kernel_h; ++ky) {
            for (int kx = 0; kx < kernel_w; ++kx) {
                // Calculate input position
                int y_in = (y_out + padding - ky * dilation - output_padding);
                int x_in = (x_out + padding - kx * dilation - output_padding);

                // Check if it's a valid input position
                if (y_in >= 0 && y_in < height * stride && y_in % stride == 0 &&
                    x_in >= 0 && x_in < width * stride && x_in % stride == 0) {
                    int y_in_s = y_in / stride;
                    int x_in_s = x_in / stride;

                    if (y_in_s < height && x_in_s < width) {
                        int input_idx = n * (in_channels * height * width) +
                                        c_in * (height * width) +
                                        y_in_s * width + x_in_s;
                        int weight_idx = c_in_idx * (out_channels * kernel_h * kernel_w) +
                                         c_out * (kernel_h * kernel_w) +
                                         ky * kernel_w + kx;

                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    scalar_t conv_bias_val = (conv_bias != nullptr) ? conv_bias[c_out] : 0.0;
    sum += conv_bias_val;

    scalar_t bias_sub = bias_tensor[c_out];
    scalar_t val = sum - bias_sub;
    
    // tanh implementation
    output[tid] = tanh(val);
}

void fused_conv_transpose_tanh_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor bias_tensor,
    torch::Tensor output,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    int out_height,
    int out_width
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    int total_threads = batch_size * output.size(1) * out_height * out_width;
    int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_conv_transpose_tanh_kernel", ([&] {
        fused_conv_transpose_tanh_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            conv_bias.defined() ? conv_bias.data_ptr<scalar_t>() : nullptr,
            bias_tensor.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            output.size(1),
            height,
            width,
            weight.size(2),
            stride,
            padding,
            output_padding,
            groups,
            dilation,
            out_height,
            out_width
        );
    }));
}
"""

# --- C++ Interface Binding ---
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_tanh_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor bias_tensor,
    torch::Tensor output,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    int out_height,
    int out_width
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_conv_transpose_tanh_forward, "Fused ConvTranspose + Bias Sub + Tanh");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

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
):
    # Compute output spatial dimensions
    kernel_size = conv_transpose_weight.size(2)
    out_h = (x.size(2) - 1) * conv_transpose_stride + (kernel_size - 1) * conv_transpose_dilation + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    out_w = (x.size(3) - 1) * conv_transpose_stride + (kernel_size - 1) * conv_transpose_dilation + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding

    output = torch.empty((x.size(0), conv_transpose_weight.size(1), out_h, out_w), device=x.device, dtype=x.dtype)

    fused_ext.fused_op(
        x.contiguous(),
        conv_transpose_weight.contiguous(),
        conv_transpose_bias.contiguous() if conv_transpose_bias is not None else torch.tensor([], device=x.device),
        bias.contiguous(),
        output,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_groups,
        conv_transpose_dilation,
        out_h,
        out_w
    )
    return output

# --- Test Setup ---
batch_size = 32
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]

# --- Example Usage ---
if __name__ == "__main__":
    x = torch.rand(batch_size, in_channels, height, width, device='cuda')
    weight = torch.rand(in_channels, out_channels // 1, kernel_size, kernel_size, device='cuda')
    conv_bias = torch.rand(out_channels, device='cuda')
    bias = torch.rand(out_channels, 1, 1, device='cuda')

    with torch.no_grad():
        out = functional_model(
            x,
            conv_transpose_weight=weight,
            conv_transpose_bias=conv_bias,
            conv_transpose_stride=1,
            conv_transpose_padding=0,
            conv_transpose_output_padding=0,
            conv_transpose_groups=1,
            conv_transpose_dilation=1,
            bias=bias
        )
        print("Output shape:", out.shape)
