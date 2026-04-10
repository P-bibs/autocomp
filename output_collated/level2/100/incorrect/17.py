# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_122448/code_1.py
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

# --- CUDA Kernel for fused ConvTranspose3d + Clamp + Div ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

template <typename scalar_t>
__global__ void fused_conv_transpose3d_clamp_div_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation,
    const scalar_t min_value,
    const scalar_t divisor,
    const int out_depth,
    const int out_height,
    const int out_width
) {
    // Output tensor indexing: [batch, channel, depth, height, width]
    int out_ch = blockIdx.x;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    int total_elements_per_channel = out_depth * out_height * out_width;

    if (out_ch >= out_channels) return;

    // Each thread block processes one output channel
    // Use shared memory for partial sums if needed, but for this case we compute directly
    for (int idx = tid; idx < total_elements_per_channel; idx += total_threads) {
        int d_out_id = idx / (out_height * out_width);
        int h_out_id = (idx / out_width) % out_height;
        int w_out_id = idx % out_width;

        scalar_t accumulator = 0.0;

        // Iterate over input channels (based on group)
        int group_id = out_ch / (out_channels / groups);
        int in_ch_start = group_id * (in_channels / groups);
        int in_ch_end = (group_id + 1) * (in_channels / groups);

        for (int in_ch = in_ch_start; in_ch < in_ch_end; ++in_ch) {
            // Kernel spatial dimensions
            for (int kd = 0; kd < kernel_size; ++kd) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        // Map output coordinate to input receptive field
                        int d_in_id = d_out_id * stride - padding + kd * dilation;
                        int h_in_id = h_out_id * stride - padding + kh * dilation;
                        int w_in_id = w_out_id * stride - padding + kw * dilation;

                        if (d_in_id >= 0 && d_in_id < in_depth &&
                            h_in_id >= 0 && h_in_id < in_height &&
                            w_in_id >= 0 && w_in_id < in_width) {

                            // Input index (NCDHW)
                            int input_idx = ((/*batch*/0 * in_channels + in_ch) * in_depth + d_in_id) * in_height * in_width +
                                            h_in_id * in_width + w_in_id;

                            // Weight index (CDHW), adjusted for group convolution
                            int weight_idx = ((out_ch * in_channels / groups + (in_ch - in_ch_start)) * kernel_size + kd) * kernel_size * kernel_size +
                                             kh * kernel_size + kw;

                            accumulator += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }

        // Add bias
        accumulator += bias[out_ch];

        // Apply clamp and division
        if (accumulator < min_value) {
            accumulator = min_value;
        }
        accumulator = accumulator / divisor;

        // Write to global memory
        int output_idx = ((/*batch*/0 * out_channels + out_ch) * out_depth + d_out_id) * out_height * out_width +
                         h_out_id * out_width + w_out_id;
        output[output_idx] = accumulator;
    }
}

void launch_fused_kernel(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation,
    const float min_value,
    const float divisor
) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_depth = input.size(2);
    const auto in_height = input.size(3);
    const auto in_width = input.size(4);

    const auto out_channels = weight.size(0);
    const auto out_depth = output.size(2);
    const auto out_height = output.size(3);
    const auto out_width = output.size(4);

    // Ensure contiguous memory layout
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();
    auto output_contig = output.contiguous();

    const dim3 threads(256);
    const dim3 blocks(out_channels);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_conv_transpose3d_clamp_div_kernel", ([&] {
        fused_conv_transpose3d_clamp_div_kernel<scalar_t><<<blocks, threads>>>(
            input_contig.data_ptr<scalar_t>(),
            weight_contig.data_ptr<scalar_t>(),
            bias_contig.data_ptr<scalar_t>(),
            output_contig.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            in_depth,
            in_height,
            in_width,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            dilation,
            static_cast<scalar_t>(min_value),
            static_cast<scalar_t>(divisor),
            out_depth,
            out_height,
            out_width
        );
    }));

    // Synchronize to catch potential errors
    cudaDeviceSynchronize();
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void launch_fused_kernel(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation,
    const float min_value,
    const float divisor
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_fused_kernel, "Fused ConvTranspose3d + Clamp + Div");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose3d_op',
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
    min_value,
    divisor,
):
    # Compute output dimensions manually
    in_depth, in_height, in_width = x.shape[-3], x.shape[-2], x.shape[-1]
    kernel_size = conv_transpose_weight.shape[-1] # assuming cubic kernel
    out_channels = conv_transpose_weight.shape[0]
    
    out_depth = (in_depth - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + conv_transpose_dilation[0] * (kernel_size - 1) + conv_transpose_output_padding[0] + 1
    out_height = (in_height - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + conv_transpose_dilation[1] * (kernel_size - 1) + conv_transpose_output_padding[1] + 1
    out_width = (in_width - 1) * conv_transpose_stride[2] - 2 * conv_transpose_padding[2] + conv_transpose_dilation[2] * (kernel_size - 1) + conv_transpose_output_padding[2] + 1

    # Allocate output tensor
    output_shape = (x.size(0), out_channels, out_depth, out_height, out_width)
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Launch fused kernel
    fused_ext.fused_op(
        x, conv_transpose_weight, conv_transpose_bias, output,
        kernel_size, conv_transpose_stride[0], conv_transpose_padding[0],
        conv_transpose_output_padding[0], conv_transpose_groups, conv_transpose_dilation[0],
        min_value, divisor
    )
    
    return output


# Configuration for evaluation
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
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width, device='cuda')]
