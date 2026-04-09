# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_045050/code_8.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x', 'add_input']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
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

# CUDA kernel source
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_conv_transpose_add_hardswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w,
    int groups,
    int dilation_d, int dilation_h, int dilation_w
) {
    // Global index for output element
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_idx >= batch_size * out_channels * out_d * out_h * out_w) return;
    
    // Decompose linear index
    int remaining = out_idx;
    int b = remaining / (out_channels * out_d * out_h * out_w);
    remaining %= (out_channels * out_d * out_h * out_w);
    int oc = remaining / (out_d * out_h * out_w);
    remaining %= (out_d * out_h * out_w);
    int od = remaining / (out_h * out_w);
    remaining %= (out_h * out_w);
    int oh = remaining / out_w;
    int ow = remaining % out_w;
    
    // Compute convolution transpose sum
    float sum = 0.0f;
    if (bias != nullptr) {
        sum = bias[oc];
    }
    
    int group_id = oc / (out_channels / groups);
    int ic_start = group_id * (in_channels / groups);
    int ic_end = ic_start + (in_channels / groups);
    
    // Iterate over input spatial dimensions and kernel
    for (int kd = 0; kd < kernel_d; kd++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int id = od - out_pad_d - kd * dilation_d + pad_d * (stride_d - 1);
                int ih = oh - out_pad_h - kh * dilation_h + pad_h * (stride_h - 1);
                int iw = ow - out_pad_w - kw * dilation_w + pad_w * (stride_w - 1);
                
                if (id % stride_d == 0 && ih % stride_h == 0 && iw % stride_w == 0) {
                    id /= stride_d;
                    ih /= stride_h;
                    iw /= stride_w;
                    
                    if (id >= 0 && id < in_d && ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        int weight_offset = oc * (in_channels / groups) * kernel_d * kernel_h * kernel_w;
                        
                        for (int ic = ic_start; ic < ic_end; ic++) {
                            int input_idx = b * in_channels * in_d * in_h * in_w + 
                                           ic * in_d * in_h * in_w + 
                                           id * in_h * in_w + 
                                           ih * in_w + iw;
                            
                            int weight_idx = weight_offset + 
                                            (ic - ic_start) * kernel_d * kernel_h * kernel_w + 
                                            kd * kernel_h * kernel_w + 
                                            kh * kernel_w + kw;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add the add_input contribution
    int add_input_idx = b * out_channels * out_d * out_h * out_w + 
                        oc * out_d * out_h * out_w + 
                        od * out_h * out_w + 
                        oh * out_w + ow;
    sum += add_input[add_input_idx];
    
    // Apply fused hardswish and multiplication: x * hardswish(x)
    // hardswish(x) = x * relu6(x + 3) / 6
    float hardswish_val = sum * fminf(fmaxf(sum + 3.0f, 0.0f), 6.0f) / 6.0f;
    float result = sum * hardswish_val;
    
    output[out_idx] = result;
}

void fused_conv_transpose_add_hardswish_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor add_input,
    torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w,
    int groups,
    int dilation_d, int dilation_h, int dilation_w
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = output.size(1);
    int in_d = input.size(2), in_h = input.size(3), in_w = input.size(4);
    int out_d = output.size(2), out_h = output.size(3), out_w = output.size(4);
    int kernel_d = weight.size(2), kernel_h = weight.size(3), kernel_w = weight.size(4);
    
    int total_elements = batch_size * out_channels * out_d * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_conv_transpose_add_hardswish_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        add_input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w, out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_pad_d, out_pad_h, out_pad_w,
        groups, dilation_d, dilation_h, dilation_w
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_add_hardswish_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor add_input,
    torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w,
    int groups,
    int dilation_d, int dilation_h, int dilation_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_add_hardswish", 
          &fused_conv_transpose_add_hardswish_forward,
          "Fused conv_transpose3d + add + hardswish kernel");
}
"""

try:
    fused_ext = load_inline(
        name='fused_conv_transpose_add_hardswish',
        cpp_sources=cpp_source,
        cuda_sources=cuda_kernel,
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        with_cuda=True
    )
except:
    fused_ext = None

def functional_model(
    x,
    add_input,
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
    """
    Fused operation: conv_transpose3d + add + hardswish activation.
    Uses custom CUDA kernel to merge all operations into a single kernel launch.
    """
    
    batch_size = x.size(0)
    in_channels = x.size(1)
    out_channels = conv_transpose_weight.size(1)
    
    # Compute output shape
    in_d, in_h, in_w = x.size(2), x.size(3), x.size(4)
    kernel_d, kernel_h, kernel_w = conv_transpose_weight.size(2), conv_transpose_weight.size(3), conv_transpose_weight.size(4)
    
    out_d = (in_d - 1) * conv_transpose_stride[0] + kernel_d - 2 * conv_transpose_padding[0] + conv_transpose_output_padding[0]
    out_h = (in_h - 1) * conv_transpose_stride[1] + kernel_h - 2 * conv_transpose_padding[1] + conv_transpose_output_padding[1]
    out_w = (in_w - 1) * conv_transpose_stride[2] + kernel_w - 2 * conv_transpose_padding[2] + conv_transpose_output_padding[2]
    
    output = torch.empty(batch_size, out_channels, out_d, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Call fused kernel
    fused_ext.fused_conv_transpose_add_hardswish(
        x.contiguous(),
        conv_transpose_weight.contiguous(),
        conv_transpose_bias.contiguous() if conv_transpose_bias is not None else torch.tensor([], device=x.device),
        add_input.contiguous(),
        output,
        conv_transpose_stride[0], conv_transpose_stride[1], conv_transpose_stride[2],
        conv_transpose_padding[0], conv_transpose_padding[1], conv_transpose_padding[2],
        conv_transpose_output_padding[0], conv_transpose_output_padding[1], conv_transpose_output_padding[2],
        conv_transpose_groups,
        conv_transpose_dilation[0], conv_transpose_dilation[1], conv_transpose_dilation[2]
    )
    
    return output


batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W), torch.rand(batch_size, out_channels, D*stride, H*stride, W*stride)]
