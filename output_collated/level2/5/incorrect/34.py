# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_115905/code_8.py
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

# CUDA kernel for fused conv_transpose2d + bias subtraction + tanh
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_conv_transpose_bias_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ subtract_bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_h,
    int input_w,
    int output_h,
    int output_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int out_pad_h,
    int out_pad_w,
    int groups,
    int dilation_h,
    int dilation_w
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch_size * out_channels * output_h * output_w) return;
    
    int b = out_idx / (out_channels * output_h * output_w);
    int oc = (out_idx / (output_h * output_w)) % out_channels;
    int oh = (out_idx / output_w) % output_h;
    int ow = out_idx % output_w;
    
    float acc = 0.0f;
    
    // Add conv_bias if present
    if (conv_bias != nullptr && conv_bias != 0) {
        acc = conv_bias[oc];
    }
    
    int group_id = oc / (out_channels / groups);
    int ic_start = group_id * (in_channels / groups);
    int ic_end = ic_start + (in_channels / groups);
    
    // Perform convolution transpose computation
    for (int ic = ic_start; ic < ic_end; ic++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int ih = oh * stride_h + kh * dilation_h - pad_h;
                int iw = ow * stride_w + kw * dilation_w - pad_w;
                
                if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                    int input_offset = b * (in_channels * input_h * input_w) +
                                     ic * (input_h * input_w) +
                                     ih * input_w + iw;
                    
                    int weight_offset = ((oc * in_channels/groups + (ic - ic_start)) * kernel_h + kh) * kernel_w + kw;
                    
                    acc += input[input_offset] * weight[weight_offset];
                }
            }
        }
    }
    
    // Subtract bias (broadcast from shape (out_channels, 1, 1))
    float bias_val = subtract_bias[oc];
    acc = acc - bias_val;
    
    // Apply tanh activation
    acc = tanhf(acc);
    
    output[out_idx] = acc;
}

void fused_conv_transpose_bias_tanh_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor subtract_bias,
    torch::Tensor output,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int out_pad_h,
    int out_pad_w,
    int groups,
    int dilation_h,
    int dilation_w
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);
    
    int out_channels = output.size(1);
    int output_h = output.size(2);
    int output_w = output.size(3);
    
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    
    int total_elements = batch_size * out_channels * output_h * output_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_conv_transpose_bias_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.numel() > 0 ? conv_bias.data_ptr<float>() : nullptr,
        subtract_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_h,
        input_w,
        output_h,
        output_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        out_pad_h,
        out_pad_w,
        groups,
        dilation_h,
        dilation_w
    );
    
    cudaDeviceSynchronize();
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_bias_tanh_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor subtract_bias,
    torch::Tensor output,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int out_pad_h,
    int out_pad_w,
    int groups,
    int dilation_h,
    int dilation_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward", &fused_conv_transpose_bias_tanh_forward, 
          "Fused conv_transpose2d + bias subtraction + tanh");
}
"""

fused_ext = load_inline(
    name='fused_conv_transpose_tanh',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False
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
    # Compute output shape
    batch_size = x.size(0)
    out_channels = conv_transpose_weight.size(0)
    
    # Calculate output height and width
    input_h, input_w = x.size(2), x.size(3)
    kernel_h, kernel_w = conv_transpose_weight.size(2), conv_transpose_weight.size(3)
    
    if isinstance(conv_transpose_stride, int):
        stride_h = stride_w = conv_transpose_stride
    else:
        stride_h, stride_w = conv_transpose_stride
    
    if isinstance(conv_transpose_padding, int):
        pad_h = pad_w = conv_transpose_padding
    else:
        pad_h, pad_w = conv_transpose_padding
    
    if isinstance(conv_transpose_output_padding, int):
        out_pad_h = out_pad_w = conv_transpose_output_padding
    else:
        out_pad_h, out_pad_w = conv_transpose_output_padding
    
    if isinstance(conv_transpose_dilation, int):
        dilation_h = dilation_w = conv_transpose_dilation
    else:
        dilation_h, dilation_w = conv_transpose_dilation
    
    output_h = (input_h - 1) * stride_h - 2 * pad_h + kernel_h + out_pad_h
    output_w = (input_w - 1) * stride_w - 2 * pad_w + kernel_w + out_pad_w
    
    # Create output tensor
    output = torch.zeros(batch_size, out_channels, output_h, output_w, 
                        dtype=x.dtype, device=x.device)
    
    # Call fused kernel
    fused_ext.fused_forward(
        x,
        conv_transpose_weight,
        conv_transpose_bias if conv_transpose_bias is not None else torch.empty(0, device=x.device),
        bias.squeeze(),
        output,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        out_pad_h,
        out_pad_w,
        conv_transpose_groups,
        dilation_h,
        dilation_w
    )
    
    return output


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
