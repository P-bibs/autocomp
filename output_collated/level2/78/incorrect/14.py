# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_031947/code_1.py
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

# Optimized CUDA kernel performing ConvTranspose3d + 2x MaxPool3d + Sum(dim=1) in one pass
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void fused_op_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int padding_d,
    const int padding_h,
    const int padding_w,
    const int output_padding_d,
    const int output_padding_h,
    const int output_padding_w,
    const int dilation_d,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int pool1_kd, const int pool1_kh, const int pool1_kw,
    const int pool1_stride_d, const int pool1_stride_h, const int pool1_stride_w,
    const int pool1_pad_d, const int pool1_pad_h, const int pool1_pad_w,
    const int pool2_kd, const int pool2_kh, const int pool2_kw,
    const int pool2_stride_d, const int pool2_stride_h, const int pool2_stride_w,
    const int pool2_pad_d, const int pool2_pad_h, const int pool2_pad_w
) {
    const int od = blockIdx.x;
    const int oh = blockIdx.y;
    const int ow = blockIdx.z;

    const int output_depth = (input_depth - 1) * stride_d - 2 * padding_d + dilation_d * (kernel_d - 1) + output_padding_d + 1;
    const int output_height = (input_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + output_padding_h + 1;
    const int output_width = (input_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + output_padding_w + 1;

    if (od >= output_depth || oh >= output_height || ow >= output_width) return;

    const int group_size = out_channels / groups;
    const int pool1_od = (od + pool1_pad_d * 2 - pool1_kd) / pool1_stride_d + 1;
    const int pool1_oh = (oh + pool1_pad_h * 2 - pool1_kh) / pool1_stride_h + 1;
    const int pool1_ow = (ow + pool1_pad_w * 2 - pool1_kw) / pool1_stride_w + 1;

    if (pool1_od < 0 || pool1_oh < 0 || pool1_ow < 0) return;

    const int pool2_od = (pool1_od + pool2_pad_d * 2 - pool2_kd) / pool2_stride_d + 1;
    const int pool2_oh = (pool1_oh + pool2_pad_h * 2 - pool2_kh) / pool2_stride_h + 1;
    const int pool2_ow = (pool1_ow + pool2_pad_w * 2 - pool2_kw) / pool2_stride_w + 1;

    if (pool2_od < 0 || pool2_oh < 0 || pool2_ow < 0) return;

    float sum_val = 0.0f;
    const int weight_c_out = out_channels / groups;

    for (int b = 0; b < batch_size; ++b) {
        float channel_max = -1e30f;
        for (int c = 0; c < out_channels; ++c) {
            float val = 0.0f;
            const int group_idx = c / group_size;
            const int weight_base_offset = group_idx * group_size * in_channels * kernel_d * kernel_h * kernel_w +
                                           (c % group_size) * in_channels * kernel_d * kernel_h * kernel_w;

            for (int kd = 0; kd < kernel_d; ++kd) {
                const int d = od + padding_d - kd * dilation_d;
                if (d < 0 || d % stride_d != 0) continue;
                const int in_d = d / stride_d;
                if (in_d >= input_depth) continue;

                for (int kh = 0; kh < kernel_h; ++kh) {
                    const int h = oh + padding_h - kh * dilation_h;
                    if (h < 0 || h % stride_h != 0) continue;
                    const int in_h = h / stride_h;
                    if (in_h >= input_height) continue;

                    for (int kw = 0; kw < kernel_w; ++kw) {
                        const int w = ow + padding_w - kw * dilation_w;
                        if (w < 0 || w % stride_w != 0) continue;
                        const int in_w = w / stride_w;
                        if (in_w >= input_width) continue;

                        for (int ic = 0; ic < in_channels; ++ic) {
                            const int input_idx = b * in_channels * input_depth * input_height * input_width +
                                                  ic * input_depth * input_height * input_width +
                                                  in_d * input_height * input_width +
                                                  in_h * input_width + in_w;
                            const int weight_idx = weight_base_offset +
                                                   ic * kernel_d * kernel_h * kernel_w +
                                                   kd * kernel_h * kernel_w +
                                                   kh * kernel_w + kw;
                            val += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            val += bias[c];

            const int pd_min = max(0, (od - pool1_kd + pool1_stride_d) / pool1_stride_d);
            const int pd_max = min(pool1_od, od / pool1_stride_d + 1);
            const int ph_min = max(0, (oh - pool1_kh + pool1_stride_h) / pool1_stride_h);
            const int ph_max = min(pool1_oh, oh / pool1_stride_h + 1);
            const int pw_min = max(0, (ow - pool1_kw + pool1_stride_w) / pool1_stride_w);
            const int pw_max = min(pool1_ow, ow / pool1_stride_w + 1);

            bool is_pool1_max = false;
            if (pd_min < pd_max && ph_min < ph_max && pw_min < pw_max) {
                for (int pd = pd_min; pd < pd_max; ++pd) {
                    for (int ph = ph_min; ph < ph_max; ++ph) {
                        for (int pw = pw_min; pw < pw_max; ++pw) {
                            const int pool_idx = pd * pool1_stride_d * pool1_stride_h * pool1_stride_w +
                                                 ph * pool1_stride_h * pool1_stride_w +
                                                 pw * pool1_stride_w;
                            const int pool_d = pd * pool1_stride_d - pool1_pad_d;
                            const int pool_h = ph * pool1_stride_h - pool1_pad_h;
                            const int pool_w = pw * pool1_stride_w - pool1_pad_w;

                            if (pool_d >= 0 && pool_d < output_depth &&
                                pool_h >= 0 && pool_h < output_height &&
                                pool_w >= 0 && pool_w < output_width) {
                                if (abs(pool_d - od) < pool1_kd &&
                                    abs(pool_h - oh) < pool1_kh &&
                                    abs(pool_w - ow) < pool1_kw) {
                                    is_pool1_max = true;
                                }
                            }
                        }
                    }
                }
            }

            if (is_pool1_max) {
                const int p2d_min = max(0, (pool1_od - pool2_kd + pool2_stride_d) / pool2_stride_d);
                const int p2d_max = min(pool2_od, pool1_od / pool2_stride_d + 1);
                const int p2h_min = max(0, (pool1_oh - pool2_kh + pool2_stride_h) / pool2_stride_h);
                const int p2h_max = min(pool2_oh, pool1_oh / pool2_stride_h + 1);
                const int p2w_min = max(0, (pool1_ow - pool2_kw + pool2_stride_w) / pool2_stride_w);
                const int p2w_max = min(pool2_ow, pool1_ow / pool2_stride_w + 1);

                bool is_pool2_max = false;
                if (p2d_min < p2d_max && p2h_min < p2h_max && p2w_min < p2w_max) {
                    for (int p2d = p2d_min; p2d < p2d_max; ++p2d) {
                        for (int p2h = p2h_min; p2h < p2h_max; ++p2h) {
                            for (int p2w = p2w_min; p2w < p2w_max; ++p2w) {
                                const int pool_d = p2d * pool2_stride_d - pool2_pad_d;
                                const int pool_h = p2h * pool2_stride_h - pool2_pad_h;
                                const int pool_w = p2w * pool2_stride_w - pool2_pad_w;

                                if (pool_d >= 0 && pool_d < output_depth &&
                                    pool_h >= 0 && pool_h < output_height &&
                                    pool_w >= 0 && pool_w < output_width) {
                                    if (abs(pool_d - pool1_od) < pool2_kd &&
                                        abs(pool_h - pool1_oh) < pool2_kh &&
                                        abs(pool_w - pool1_ow) < pool2_kw) {
                                        is_pool2_max = true;
                                    }
                                }
                            }
                        }
                    }
                }

                if (is_pool2_max) {
                    if (val > channel_max) {
                        channel_max = val;
                    }
                }
            }
        }
        sum_val += channel_max;
    }

    const int out_idx = pool2_od * ((output_height - pool1_kh) / pool1_stride_h + 1) * ((output_width - pool1_kw) / pool1_stride_w + 1) +
                        pool2_oh * ((output_width - pool1_kw) / pool1_stride_w + 1) +
                        pool2_ow;
    output[out_idx] = sum_val;
}

void fused_op_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int padding_d, const int padding_h, const int padding_w,
    const int output_padding_d, const int output_padding_h, const int output_padding_w,
    const int dilation_d, const int dilation_h, const int dilation_w,
    const int groups,
    const int pool1_kd, const int pool1_kh, const int pool1_kw,
    const int pool1_stride_d, const int pool1_stride_h, const int pool1_stride_w,
    const int pool1_pad_d, const int pool1_pad_h, const int pool1_pad_w,
    const int pool2_kd, const int pool2_kh, const int pool2_kw,
    const int pool2_stride_d, const int pool2_stride_h, const int pool2_stride_w,
    const int pool2_pad_d, const int pool2_pad_h, const int pool2_pad_w
) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_depth = input.size(2);
    const auto input_height = input.size(3);
    const auto input_width = input.size(4);
    const auto out_channels = weight.size(0);

    const int output_depth = (input_depth - 1) * stride_d - 2 * padding_d + dilation_d * (kernel_d - 1) + output_padding_d + 1;
    const int output_height = (input_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + output_padding_h + 1;
    const int output_width = (input_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + output_padding_w + 1;

    const dim3 grid(output_depth, output_height, output_width);
    const dim3 block(1, 1, 1);

    const at::cuda::CUDAGuard device_guard(input.device());
    fused_op_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        dilation_d, dilation_h, dilation_w,
        groups,
        pool1_kd, pool1_kh, pool1_kw,
        pool1_stride_d, pool1_stride_h, pool1_stride_w,
        pool1_pad_d, pool1_pad_h, pool1_pad_w,
        pool2_kd, pool2_kh, pool2_kw,
        pool2_stride_d, pool2_stride_h, pool2_stride_w,
        pool2_pad_d, pool2_pad_h, pool2_pad_w
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int padding_d, const int padding_h, const int padding_w,
    const int output_padding_d, const int output_padding_h, const int output_padding_w,
    const int dilation_d, const int dilation_h, const int dilation_w,
    const int groups,
    const int pool1_kd, const int pool1_kh, const int pool1_kw,
    const int pool1_stride_d, const int pool1_stride_h, const int pool1_stride_w,
    const int pool1_pad_d, const int pool1_pad_h, const int pool1_pad_w,
    const int pool2_kd, const int pool2_kh, const int pool2_kw,
    const int pool2_stride_d, const int pool2_stride_h, const int pool2_stride_w,
    const int pool2_pad_d, const int pool2_pad_h, const int pool2_pad_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused ConvTranspose3d + 2x MaxPool3d + Sum");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, 
                     conv_transpose_stride, conv_transpose_padding, 
                     conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, max_pool1_kernel_size, 
                     max_pool1_stride, max_pool1_padding, max_pool1_dilation, 
                     max_pool1_ceil_mode, max_pool1_return_indices, 
                     max_pool2_kernel_size, max_pool2_stride, 
                     max_pool2_padding, max_pool2_dilation, 
                     max_pool2_ceil_mode, max_pool2_return_indices):
    batch_size = x.size(0)
    in_channels = x.size(1)
    out_channels = conv_transpose_weight.size(0)
    
    kernel_d, kernel_h, kernel_w = conv_transpose_weight.shape[2], conv_transpose_weight.shape[3], conv_transpose_weight.shape[4]
    stride_d, stride_h, stride_w = conv_transpose_stride
    padding_d, padding_h, padding_w = conv_transpose_padding
    output_padding_d, output_padding_h, output_padding_w = conv_transpose_output_padding
    dilation_d, dilation_h, dilation_w = conv_transpose_dilation
    
    pool1_kd, pool1_kh, pool1_kw = max_pool1_kernel_size
    pool1_stride_d, pool1_stride_h, pool1_stride_w = max_pool1_stride
    pool1_pad_d, pool1_pad_h, pool1_pad_w = max_pool1_padding
    
    pool2_kd, pool2_kh, pool2_kw = max_pool2_kernel_size
    pool2_stride_d, pool2_stride_h, pool2_stride_w = max_pool2_stride
    pool2_pad_d, pool2_pad_h, pool2_pad_w = max_pool2_padding
    
    # Compute final output shape after two pooling operations
    conv_out_d = (x.size(2) - 1) * stride_d - 2 * padding_d + dilation_d * (kernel_d - 1) + output_padding_d + 1
    conv_out_h = (x.size(3) - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + output_padding_h + 1
    conv_out_w = (x.size(4) - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + output_padding_w + 1
    
    pool1_out_d = (conv_out_d + 2 * pool1_pad_d - pool1_kd) // pool1_stride_d + 1
    pool1_out_h = (conv_out_h + 2 * pool1_pad_h - pool1_kh) // pool1_stride_h + 1
    pool1_out_w = (conv_out_w + 2 * pool1_pad_w - pool1_kw) // pool1_stride_w + 1
    
    pool2_out_d = (pool1_out_d + 2 * pool2_pad_d - pool2_kd) // pool2_stride_d + 1
    pool2_out_h = (pool1_out_h + 2 * pool2_pad_h - pool2_kh) // pool2_stride_h + 1
    pool2_out_w = (pool1_out_w + 2 * pool2_pad_w - pool2_kw) // pool2_stride_w + 1
    
    output_shape = (batch_size, 1, pool2_out_d, pool2_out_h, pool2_out_w)
    output = torch.zeros(output_shape, device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(
        x, conv_transpose_weight, conv_transpose_bias, output,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        dilation_d, dilation_h, dilation_w,
        conv_transpose_groups,
        pool1_kd, pool1_kh, pool1_kw,
        pool1_stride_d, pool1_stride_h, pool1_stride_w,
        pool1_pad_d, pool1_pad_h, pool1_pad_w,
        pool2_kd, pool2_kh, pool2_kw,
        pool2_stride_d, pool2_stride_h, pool2_stride_w,
        pool2_pad_d, pool2_pad_h, pool2_pad_w
    )
    
    return output

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
