# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_130922/code_11.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, followed by a sum, 
    a residual add, a multiplication, and another residual add.
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Fused 3D transposed convolution + post-processing kernel
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_transpose_postproc_kernel(
    const float* input,
    const float* weight,
    const float* conv_bias,
    const float* post_bias,
    float* output,
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,
    int64_t in_d, int64_t in_h, int64_t in_w,
    int64_t out_d, int64_t out_h, int64_t out_w,
    int64_t kernel_d, int64_t kernel_h, int64_t kernel_w,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t pad_d, int64_t pad_h, int64_t pad_w,
    int64_t out_pad_d, int64_t out_pad_h, int64_t out_pad_w
) {
    // Each thread computes one output element
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_idx >= batch_size * out_channels * out_d * out_h * out_w) {
        return;
    }
    
    // Decode output position
    int64_t remaining = out_idx;
    int64_t b = remaining / (out_channels * out_d * out_h * out_w);
    remaining %= (out_channels * out_d * out_h * out_w);
    int64_t oc = remaining / (out_d * out_h * out_w);
    remaining %= (out_d * out_h * out_w);
    int64_t od = remaining / (out_h * out_w);
    remaining %= (out_h * out_w);
    int64_t oh = remaining / out_w;
    int64_t ow = remaining % out_w;
    
    // Accumulate convolution result
    float conv_result = (conv_bias != nullptr) ? conv_bias[oc] : 0.0f;
    
    // Iterate over input positions that contribute to this output position
    for (int64_t kd = 0; kd < kernel_d; kd++) {
        for (int64_t kh = 0; kh < kernel_h; kh++) {
            for (int64_t kw = 0; kw < kernel_w; kw++) {
                // Transposed convolution: map output to input
                int64_t id = (od - out_pad_d - kd + pad_d);
                int64_t ih = (oh - out_pad_h - kh + pad_h);
                int64_t iw = (ow - out_pad_w - kw + pad_w);
                
                if (id >= 0 && id < in_d * stride_d && ih >= 0 && ih < in_h * stride_h && 
                    iw >= 0 && iw < in_w * stride_w) {
                    if (id % stride_d == 0 && ih % stride_h == 0 && iw % stride_w == 0) {
                        int64_t in_d_idx = id / stride_d;
                        int64_t in_h_idx = ih / stride_h;
                        int64_t in_w_idx = iw / stride_w;
                        
                        // Compute weight index
                        int64_t w_idx = oc * (in_channels * kernel_d * kernel_h * kernel_w) +
                                       0 * (kernel_d * kernel_h * kernel_w) +
                                       kd * (kernel_h * kernel_w) +
                                       kh * kernel_w +
                                       kw;
                        
                        for (int64_t ic = 0; ic < in_channels; ic++) {
                            int64_t in_idx = b * (in_channels * in_d * in_h * in_w) +
                                           ic * (in_d * in_h * in_w) +
                                           in_d_idx * (in_h * in_w) +
                                           in_h_idx * in_w +
                                           in_w_idx;
                            
                            int64_t weight_idx = w_idx + ic * (kernel_d * kernel_h * kernel_w);
                            
                            conv_result += input[in_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Apply post-processing in same kernel: ((x + bias) + x) * x + x
    float x_val = conv_result;
    float b_val = (post_bias != nullptr) ? post_bias[oc] : 0.0f;
    float result = ((x_val + b_val) + x_val) * x_val + x_val;
    
    output[out_idx] = result;
}

void fused_conv_transpose_postproc_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef output_padding,
    int64_t out_d, int64_t out_h, int64_t out_w
) {
    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t out_channels = weight.size(1);
    int64_t in_d = input.size(2);
    int64_t in_h = input.size(3);
    int64_t in_w = input.size(4);
    
    int64_t kernel_d = weight.size(2);
    int64_t kernel_h = weight.size(3);
    int64_t kernel_w = weight.size(4);
    
    int64_t num_elements = batch_size * out_channels * out_d * out_h * out_w;
    int64_t threads_per_block = 256;
    int64_t blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    
    fused_conv_transpose_postproc_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.defined() ? conv_bias.data_ptr<float>() : nullptr,
        post_bias.defined() ? post_bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w, out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2]
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_postproc_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef output_padding,
    int64_t out_d, int64_t out_h, int64_t out_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_postproc", &fused_conv_transpose_postproc_forward, 
          "Fused 3D transposed convolution with post-processing");
}
"""

fused_ext = load_inline(
    name='fused_conv_transpose_postproc_ext',
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
    out_channels = conv_transpose_weight.size(1)
    
    out_d = (x.size(2) - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + conv_transpose_weight.size(2) + conv_transpose_output_padding[0]
    out_h = (x.size(3) - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + conv_transpose_weight.size(3) + conv_transpose_output_padding[1]
    out_w = (x.size(4) - 1) * conv_transpose_stride[2] - 2 * conv_transpose_padding[2] + conv_transpose_weight.size(4) + conv_transpose_output_padding[2]
    
    output = torch.empty(batch_size, out_channels, out_d, out_h, out_w, dtype=x.dtype, device=x.device)
    
    bias_flat = bias.view(-1) if bias is not None else None
    
    # Call fused kernel that does both convolution and post-processing
    fused_ext.fused_conv_transpose_postproc(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        bias_flat,
        output,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        out_d, out_h, out_w
    )
    
    return output

batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
