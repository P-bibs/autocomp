# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_112831/code_2.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- CUDA Kernel for fused conv_transpose2d + bias subtraction + tanh ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

template <typename scalar_t>
__global__ void fused_conv_transpose2d_tanh_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ conv_bias,
    const scalar_t* __restrict__ post_bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * output_height * output_width;

    if (tid >= total_elements) return;

    const int n = tid / (out_channels * output_height * output_width);
    const int c_out = (tid / (output_height * output_width)) % out_channels;
    const int h_out = (tid / output_width) % output_height;
    const int w_out = tid % output_width;

    scalar_t sum = 0.0;

    const int group_id = c_out / (out_channels / groups);
    const int weight_offset = group_id * (in_channels / groups) * out_channels * kernel_size * kernel_size;

    for (int c_in_group = 0; c_in_group < in_channels / groups; c_in_group++) {
        const int c_in = group_id * (in_channels / groups) + c_in_group;

        const int h_start = h_out * stride - padding;
        const int w_start = w_out * stride - padding;

        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                const int h_in = h_start + dilation * kh;
                const int w_in = w_start + dilation * kw;

                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    const int input_idx = n * (in_channels * input_height * input_width) +
                                          c_in * (input_height * input_width) +
                                          h_in * input_width + w_in;

                    const int weight_idx = weight_offset +
                                           c_in_group * (out_channels * kernel_size * kernel_size) +
                                           c_out * (kernel_size * kernel_size) +
                                           kh * kernel_size + kw;

                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    if (conv_bias != nullptr) {
        sum += conv_bias[c_out];
    }

    sum -= post_bias[c_out];
    output[tid] = tanh(sum);
}

void fused_conv_transpose2d_tanh(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& conv_bias,
    const at::Tensor& post_bias,
    at::Tensor& output,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation
) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);
    
    const auto out_channels = weight.size(1);
    const auto kernel_size = weight.size(2);
    
    const auto output_height = (input_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    const auto output_width = (input_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;

    const auto total_elements = batch_size * out_channels * output_height * output_width;
    const int threads = 512;
    const int blocks = (total_elements + threads - 1) / threads;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "fused_conv_transpose2d_tanh_kernel", ([&] {
        fused_conv_transpose2d_tanh_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            conv_bias.has_value() ? conv_bias.value().data_ptr<scalar_t>() : nullptr,
            post_bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            dilation
        );
    }));
    
    cudaDeviceSynchronize();
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose2d_tanh(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& conv_bias,
    const at::Tensor& post_bias,
    at::Tensor& output,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose2d_tanh", &fused_conv_transpose2d_tanh, "Fused conv_transpose2d + bias subtraction + tanh");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose2d_tanh_ext',
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
    # Calculate output dimensions
    batch_size = x.size(0)
    in_channels = x.size(1)
    input_height = x.size(2)
    input_width = x.size(3)
    
    out_channels = conv_transpose_weight.size(1)
    kernel_size = conv_transpose_weight.size(2)
    
    output_height = (input_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + conv_transpose_output_padding + 1
    output_width = (input_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + conv_transpose_output_padding + 1
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, output_height, output_width), dtype=x.dtype, device=x.device)
    
    # Call fused kernel
    fused_ext.fused_conv_transpose2d_tanh(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        bias,
        output,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_groups,
        conv_transpose_dilation
    )
    
    return output

batch_size = 32
in_channels  = 64  
out_channels = 64  
height = width = 256 
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
