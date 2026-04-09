# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_034836/code_2.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_transpose_pool_sum_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int pool1_kernel_d, int pool1_kernel_h, int pool1_kernel_w,
    int pool1_stride_d, int pool1_stride_h, int pool1_stride_w,
    int pool1_padding_d, int pool1_padding_h, int pool1_padding_w,
    int pool2_kernel_d, int pool2_kernel_h, int pool2_kernel_w,
    int pool2_stride_d, int pool2_stride_h, int pool2_stride_w,
    int pool2_padding_d, int pool2_padding_h, int pool2_padding_w,
    int pool1_out_d, int pool1_out_h, int pool1_out_w,
    int final_out_d, int final_out_h, int final_out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Output dimensions: (batch, 1, final_d, final_h, final_w)
    int total_output_size = batch_size * final_out_d * final_out_h * final_out_w;
    
    if (idx >= total_output_size) return;
    
    // Decode output index
    int b = idx / (final_out_d * final_out_h * final_out_w);
    int rem = idx % (final_out_d * final_out_h * final_out_w);
    int od = rem / (final_out_h * final_out_w);
    rem = rem % (final_out_h * final_out_w);
    int oh = rem / final_out_w;
    int ow = rem % final_out_w;
    
    // Backward map through pool2 to pool1 output space
    int pool1_od_start = od * pool2_stride_d - pool2_padding_d;
    int pool1_oh_start = oh * pool2_stride_h - pool2_padding_h;
    int pool1_ow_start = ow * pool2_stride_w - pool2_padding_w;
    
    float result = 0.0f;
    
    // Iterate over pool2 kernel
    for (int kd2 = 0; kd2 < pool2_kernel_d; kd2++) {
        for (int kh2 = 0; kh2 < pool2_kernel_h; kh2++) {
            for (int kw2 = 0; kw2 < pool2_kernel_w; kw2++) {
                int pool1_od = pool1_od_start + kd2;
                int pool1_oh = pool1_oh_start + kh2;
                int pool1_ow = pool1_ow_start + kw2;
                
                if (pool1_od < 0 || pool1_od >= pool1_out_d ||
                    pool1_oh < 0 || pool1_oh >= pool1_out_h ||
                    pool1_ow < 0 || pool1_ow >= pool1_out_w) {
                    continue;
                }
                
                // Now compute pool1 output value at (pool1_od, pool1_oh, pool1_ow)
                // Backward map through pool1 to conv_transpose output space
                int conv_od_start = pool1_od * pool1_stride_d - pool1_padding_d;
                int conv_oh_start = pool1_oh * pool1_stride_h - pool1_padding_h;
                int conv_ow_start = pool1_ow * pool1_stride_w - pool1_padding_w;
                
                float pool1_max = -1e9f;
                
                // Pool1 kernel
                for (int kd1 = 0; kd1 < pool1_kernel_d; kd1++) {
                    for (int kh1 = 0; kh1 < pool1_kernel_h; kh1++) {
                        for (int kw1 = 0; kw1 < pool1_kernel_w; kw1++) {
                            int conv_od = conv_od_start + kd1;
                            int conv_oh = conv_oh_start + kh1;
                            int conv_ow = conv_ow_start + kw1;
                            
                            if (conv_od < 0 || conv_od >= out_d ||
                                conv_oh < 0 || conv_oh >= out_h ||
                                conv_ow < 0 || conv_ow >= out_w) {
                                continue;
                            }
                            
                            // Compute conv_transpose output at (b, c, conv_od, conv_oh, conv_ow)
                            float conv_val = 0.0f;
                            
                            for (int c = 0; c < out_channels; c++) {
                                float chan_val = (bias != nullptr) ? bias[c] : 0.0f;
                                
                                for (int ic = 0; ic < in_channels; ic++) {
                                    for (int kd = 0; kd < kernel_d; kd++) {
                                        for (int kh = 0; kh < kernel_h; kh++) {
                                            for (int kw = 0; kw < kernel_w; kw++) {
                                                int id = conv_od - output_padding_d + padding_d - dilation_d * (kernel_d - 1 - kd);
                                                int ih = conv_oh - output_padding_h + padding_h - dilation_h * (kernel_h - 1 - kh);
                                                int iw = conv_ow - output_padding_w + padding_w - dilation_w * (kernel_w - 1 - kw);
                                                
                                                if (id % stride_d == 0 && ih % stride_h == 0 && iw % stride_w == 0) {
                                                    id /= stride_d;
                                                    ih /= stride_h;
                                                    iw /= stride_w;
                                                    
                                                    if (id >= 0 && id < in_d && ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                                        int inp_idx = ((b * in_channels + ic) * in_d + id) * in_h * in_w + ih * in_w + iw;
                                                        int w_idx = ((c * in_channels + ic) * kernel_d + kd) * kernel_h * kernel_w + kh * kernel_w + kw;
                                                        chan_val += input[inp_idx] * weight[w_idx];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    
                                    conv_val += chan_val;
                                }
                            }
                            
                            pool1_max = fmaxf(pool1_max, conv_val);
                        }
                    }
                }
                
                result += pool1_max;
            }
        }
    }
    
    output[idx] = result;
}

void fused_conv_transpose_pool_sum_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int pool1_kernel_d, int pool1_kernel_h, int pool1_kernel_w,
    int pool1_stride_d, int pool1_stride_h, int pool1_stride_w,
    int pool1_padding_d, int pool1_padding_h, int pool1_padding_w,
    int pool2_kernel_d, int pool2_kernel_h, int pool2_kernel_w,
    int pool2_stride_d, int pool2_stride_h, int pool2_stride_w,
    int pool2_padding_d, int pool2_padding_h, int pool2_padding_w,
    int pool1_out_d, int pool1_out_h, int pool1_out_w,
    int final_out_d, int final_out_h, int final_out_w
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_d = input.size(2);
    int in_h = input.size(3);
    int in_w = input.size(4);
    int out_channels = weight.size(0);
    int out_d = output.size(2);
    int out_h = output.size(3);
    int out_w = output.size(4);
    
    int total_output_size = batch_size * final_out_d * final_out_h * final_out_w;
    int threads = 256;
    int blocks = (total_output_size + threads - 1) / threads;
    
    fused_conv_transpose_pool_sum_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w, out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        dilation_d, dilation_h, dilation_w,
        pool1_kernel_d, pool1_kernel_h, pool1_kernel_w,
        pool1_stride_d, pool1_stride_h, pool1_stride_w,
        pool1_padding_d, pool1_padding_h, pool1_padding_w,
        pool2_kernel_d, pool2_kernel_h, pool2_kernel_w,
        pool2_stride_d, pool2_stride_h, pool2_stride_w,
        pool2_padding_d, pool2_padding_h, pool2_padding_w,
        pool1_out_d, pool1_out_h, pool1_out_w,
        final_out_d, final_out_h, final_out_w
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_pool_sum_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int pool1_kernel_d, int pool1_kernel_h, int pool1_kernel_w,
    int pool1_stride_d, int pool1_stride_h, int pool1_stride_w,
    int pool1_padding_d, int pool1_padding_h, int pool1_padding_w,
    int pool2_kernel_d, int pool2_kernel_h, int pool2_kernel_w,
    int pool2_stride_d, int pool2_stride_h, int pool2_stride_w,
    int pool2_padding_d, int pool2_padding_h, int pool2_padding_w,
    int pool1_out_d, int pool1_out_h, int pool1_out_w,
    int final_out_d, int final_out_h, int final_out_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward", &fused_conv_transpose_pool_sum_forward, "Fused conv_transpose + pool + sum");
}
"""

fused_ext = load_inline(
    name='fused_ops',
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
    batch_size = x.size(0)
    
    # Compute output dimensions manually
    in_d, in_h, in_w = x.size(2), x.size(3), x.size(4)
    
    # Conv transpose output size
    out_d = (in_d - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + conv_transpose_weight.size(2) + conv_transpose_output_padding[0]
    out_h = (in_h - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + conv_transpose_weight.size(3) + conv_transpose_output_padding[1]
    out_w = (in_w - 1) * conv_transpose_stride[2] - 2 * conv_transpose_padding[2] + conv_transpose_weight.size(4) + conv_transpose_output_padding[2]
    
    # Pool1 output size
    pool1_out_d = (out_d + 2 * max_pool1_padding[0] - max_pool1_dilation[0] * (max_pool1_kernel_size[0] - 1) - 1) // max_pool1_stride[0] + 1
    pool1_out_h = (out_h + 2 * max_pool1_padding[1] - max_pool1_dilation[1] * (max_pool1_kernel_size[1] - 1) - 1) // max_pool1_stride[1] + 1
    pool1_out_w = (out_w + 2 * max_pool1_padding[2] - max_pool1_dilation[2] * (max_pool1_kernel_size[2] - 1) - 1) // max_pool1_stride[2] + 1
    
    # Pool2 output size
    final_out_d = (pool1_out_d + 2 * max_pool2_padding[0] - max_pool2_dilation[0] * (max_pool2_kernel_size[0] - 1) - 1) // max_pool2_stride[0] + 1
    final_out_h = (pool1_out_h + 2 * max_pool2_padding[1] - max_pool2_dilation[1] * (max_pool2_kernel_size[1] - 1) - 1) // max_pool2_stride[1] + 1
    final_out_w = (pool1_out_w + 2 * max_pool2_padding[2] - max_pool2_dilation[2] * (max_pool2_kernel_size[2] - 1) - 1) // max_pool2_stride[2] + 1
    
    # Allocate output tensor
    output = torch.zeros(batch_size, 1, final_out_d, final_out_h, final_out_w, 
                        dtype=x.dtype, device=x.device)
    
    # Call fused kernel
    fused_ext.fused_forward(
        x, conv_transpose_weight, conv_transpose_bias, output,
        conv_transpose_weight.size(2), conv_transpose_weight.size(3), conv_transpose_weight.size(4),
        conv_transpose_stride[0], conv_transpose_stride[1], conv_transpose_stride[2],
        conv_transpose_padding[0], conv_transpose_padding[1], conv_transpose_padding[2],
        conv_transpose_output_padding[0], conv_transpose_output_padding[1], conv_transpose_output_padding[2],
        conv_transpose_dilation[0], conv_transpose_dilation[1], conv_transpose_dilation[2],
        max_pool1_kernel_size[0], max_pool1_kernel_size[1], max_pool1_kernel_size[2],
        max_pool1_stride[0], max_pool1_stride[1], max_pool1_stride[2],
        max_pool1_padding[0], max_pool1_padding[1], max_pool1_padding[2],
        max_pool2_kernel_size[0], max_pool2_kernel_size[1], max_pool2_kernel_size[2],
        max_pool2_stride[0], max_pool2_stride[1], max_pool2_stride[2],
        max_pool2_padding[0], max_pool2_padding[1], max_pool2_padding[2],
        pool1_out_d, pool1_out_h, pool1_out_w,
        final_out_d, final_out_h, final_out_w
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
