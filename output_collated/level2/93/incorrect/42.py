# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_155151/code_4.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'add_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'add_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

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
    if 'add_value' in flat_state:
        state_kwargs['add_value'] = flat_state['add_value']
    else:
        state_kwargs['add_value'] = getattr(model, 'add_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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

# Custom CUDA implementation of fused conv_transpose2d + add + min + gelu + mul
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_transpose_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    float add_val,
    float mul_val
) {
    // Grid-stride loop to handle all output elements
    int total_out = batch_size * out_channels * out_h * out_w;
    
    for (int out_idx = blockIdx.x * blockDim.x + threadIdx.x; 
         out_idx < total_out; 
         out_idx += gridDim.x * blockDim.x) {
        
        // Decode output indices
        int b = out_idx / (out_channels * out_h * out_w);
        int c = (out_idx / (out_h * out_w)) % out_channels;
        int oh = (out_idx / out_w) % out_h;
        int ow = out_idx % out_w;
        
        // Accumulate conv_transpose2d result
        float val = 0.0f;
        
        // Add bias
        if (bias != nullptr) {
            val = bias[c];
        }
        
        // Perform transposed convolution
        // For conv_transpose2d: output[b,c,oh,ow] = sum over input spatial and in_channels
        // of input[b,ic,ih,iw] * weight[ic,c,kh,kw] where:
        // ih = (oh + pad_h - kh) / stride_h and iw = (ow + pad_w - kw) / stride_w
        
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int ih = oh - kh * stride_h + pad_h;
                    int iw = ow - kw * stride_w + pad_w;
                    
                    // Check if input indices are valid
                    if (ih >= 0 && ih < in_h * stride_h && ih % stride_h == 0 &&
                        iw >= 0 && iw < in_w * stride_w && iw % stride_w == 0) {
                        
                        ih /= stride_h;
                        iw /= stride_w;
                        
                        if (ih < in_h && iw < in_w) {
                            int in_idx = ((b * in_channels + ic) * in_h + ih) * in_w + iw;
                            int w_idx = ((ic * out_channels + c) * kernel_h + kh) * kernel_w + kw;
                            
                            val += input[in_idx] * weight[w_idx];
                        }
                    }
                }
            }
        }
        
        // Fused operations: add + min + gelu + mul
        val = val + add_val;
        val = fminf(val, 0.0f);
        val = fast_gelu(val);
        val = val * mul_val;
        
        output[out_idx] = val;
    }
}

void fused_conv_transpose_gelu_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    float add_val,
    float mul_val
) {
    int total_out = batch_size * out_channels * out_h * out_w;
    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    blocks = min(blocks, 65535); // Cap blocks for GPU limits
    
    float* bias_ptr = bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;
    
    fused_conv_transpose_gelu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        add_val,
        mul_val
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_gelu_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    float add_val,
    float mul_val
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_gelu", &fused_conv_transpose_gelu_forward, 
          "Fused conv_transpose2d + add + min + gelu + mul operation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose_gelu',
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
    add_value,
    multiply_value,
):
    # Validate that we're working with supported parameters
    if conv_transpose_groups != 1:
        raise ValueError("Grouped convolutions not supported")
    if conv_transpose_dilation != 1:
        raise ValueError("Dilated convolutions not supported")
        
    batch_size = x.shape[0]
    in_channels = x.shape[1]
    in_h, in_w = x.shape[2], x.shape[3]
    
    out_channels = conv_transpose_weight.shape[0]  # Note: different from original code
    kernel_h, kernel_w = conv_transpose_weight.shape[2], conv_transpose_weight.shape[3]
    
    # Handle both tuple and int stride/padding
    if isinstance(conv_transpose_stride, tuple):
        stride_h, stride_w = conv_transpose_stride
    else:
        stride_h = stride_w = conv_transpose_stride
        
    if isinstance(conv_transpose_padding, tuple):
        pad_h, pad_w = conv_transpose_padding
    else:
        pad_h = pad_w = conv_transpose_padding
    
    # Calculate output spatial dimensions for conv_transpose2d
    out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h + conv_transpose_output_padding
    out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w + conv_transpose_output_padding
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, 
                      dtype=x.dtype, device=x.device)
    
    # Call fused kernel
    fused_ext.fused_conv_transpose_gelu(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        out,
        batch_size,
        in_channels,
        out_channels,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        float(add_value),
        float(multiply_value)
    )
    
    return out

# Constants provided by original context
batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
