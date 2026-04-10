# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_162535/code_0.py
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
from torch.utils.cpp_extension import load_inline

# --- CUDA Kernel ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

template <typename scalar_t>
__global__ void fused_convtranspose2d_bias_clamp_scale_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> conv_bias,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> add_bias,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output,
    const int out_channels,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int output_padding_h,
    const int output_padding_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const scalar_t scaling_factor
) {
    int batch_idx = blockIdx.x;
    int out_ch = blockIdx.y * blockDim.x + threadIdx.x;

    if (batch_idx >= input.size(0) || out_ch >= out_channels) return;

    int OH = output.size(2);
    int OW = output.size(3);

    int IH = input.size(2);
    int IW = input.size(3);

    int G = groups;
    int group_idx = out_ch / (out_channels / G);
    int group_out_ch = out_ch % (out_channels / G);
    int in_ch_per_group = input.size(1) / G;

    scalar_t scale_inv = 1.0 / scaling_factor;

    for (int out_y = 0; out_y < OH; ++out_y) {
        for (int out_x = 0; out_x < OW; ++out_x) {
            scalar_t value = 0.0;
            for (int ky = 0; ky < kernel_h; ++ky) {
                for (int kx = 0; kx < kernel_w; ++kx) {
                    int in_y = out_y * stride_h - padding_h + ky * dilation_h;
                    int in_x = out_x * stride_w - padding_w + kx * dilation_w;
                    if (in_y >= 0 && in_y < IH && in_x >= 0 && in_x < IW) {
                        for (int ic = 0; ic < in_ch_per_group; ++ic) {
                            int in_ch = group_idx * in_ch_per_group + ic;
                            scalar_t inp_val = input[batch_idx][in_ch][in_y][in_x];
                            scalar_t wgt_val = weight[out_ch][ic][ky][kx];
                            value += inp_val * wgt_val;
                        }
                    }
                }
            }
            value += conv_bias[out_ch];
            value += add_bias[out_ch];

            // First clamp [0.0, 1.0]
            value = fmaxf(0.0f, fminf(1.0f, value));

            // Scale and clamp again
            value = value * scaling_factor;
            value = fmaxf(0.0f, fminf(1.0f, value));

            // Final scaling back
            value = value * scale_inv;

            output[batch_idx][out_ch][out_y][out_x] = value;
        }
    }
}

void fused_convtranspose2d_bias_clamp_scale(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& conv_bias,
    const at::Tensor& add_bias,
    at::Tensor& output,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int dilation_h, int dilation_w,
    int groups,
    float scaling_factor
) {
    const auto batch_size = input.size(0);
    const auto out_channels = weight.size(0);

    dim3 block(32); 
    dim3 grid(batch_size, (out_channels + block.x - 1) / block.x);

    const at::cuda::OptionalCUDAGuard device_guard(input.device());

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_convtranspose2d_bias_clamp_scale", ([&] {
        fused_convtranspose2d_bias_clamp_scale_kernel<scalar_t><<<grid, block>>>(
            input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            weight.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            conv_bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            add_bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            output_padding_h,
            output_padding_w,
            dilation_h,
            dilation_w,
            groups,
            static_cast<scalar_t>(scaling_factor)
        );
    }));
}
"""

# --- C++ Binding Code ---
cpp_source = r"""
#include <torch/extension.h>

void fused_convtranspose2d_bias_clamp_scale(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& conv_bias,
    const at::Tensor& add_bias,
    at::Tensor& output,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int dilation_h, int dilation_w,
    int groups,
    float scaling_factor
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_convtranspose2d_bias_clamp_scale, "Fused ConvTranspose2d + bias + clamp + scale");
}
"""

# --- Compile Custom Extension ---
fused_ext = load_inline(
    name="fused_op_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True
)

# --- Updated functional_model ---
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
    # Allocate output tensor
    oshape = x.shape[:1] + (conv_transpose_weight.shape[0],)
    oh = (x.shape[2] - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + conv_transpose_weight.shape[2] + conv_transpose_output_padding[0]
    ow = (x.shape[3] - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + conv_transpose_weight.shape[3] + conv_transpose_output_padding[1]
    oshape += (oh, ow)
    output = torch.empty(oshape, dtype=x.dtype, device=x.device)

    # Call fused kernel
    fused_ext.fused_op(
        x, conv_transpose_weight, conv_transpose_bias, bias.squeeze(),
        output,
        conv_transpose_weight.shape[2], conv_transpose_weight.shape[3],
        conv_transpose_stride[0], conv_transpose_stride[1],
        conv_transpose_padding[0], conv_transpose_padding[1],
        conv_transpose_output_padding[0], conv_transpose_output_padding[1],
        conv_transpose_dilation[0], conv_transpose_dilation[1],
        conv_transpose_groups,
        float(scaling_factor)
    )

    return output

batch_size = 128
in_channels = 64
out_channels = 64
height = width = 128
kernel_size = 3
stride = (2, 2)
padding = (1, 1)
output_padding = (1, 1)
bias_shape = (out_channels,)
scaling_factor = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
