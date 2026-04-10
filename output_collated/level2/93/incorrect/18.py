# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_153448/code_3.py
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
from torch.utils.cpp_extension import load_inline

# CUDA kernel: Fused Transposed Conv2d + Pointwise Ops
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fused_conv_transpose_act_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output, 
    float add_val, 
    float mul_val,
    int batch, 
    int in_c, 
    int out_c, 
    int in_h, 
    int in_w, 
    int k, 
    int stride,
    int pad,
    int out_pad,
    int dilation,
    int groups) {
    
    int out_h = (in_h - 1) * stride - 2 * pad + dilation * (k - 1) + out_pad + 1;
    int out_w = (in_w - 1) * stride - 2 * pad + dilation * (k - 1) + out_pad + 1;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch * out_c * out_h * out_w;

    if (tid < total_threads) {
        int temp = tid;
        int w_out = temp % out_w; temp /= out_w;
        int h_out = temp % out_h; temp /= out_h;
        int oc = temp % out_c; temp /= out_c;
        int b = temp;

        float acc = 0.0f;
        
        // Direct Transposed Convolution computation
        for (int ic = 0; ic < in_c; ++ic) {
            for (int kh = 0; kh < k; ++kh) {
                for (int kw = 0; kw < k; ++kw) {
                    // Calculate input coordinate
                    int h_in = h_out + pad - kh * dilation;
                    int w_in = w_out + pad - kw * dilation;
                    
                    if (h_in % stride == 0 && w_in % stride == 0) {
                        h_in /= stride;
                        w_in /= stride;
                        
                        if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                            int input_idx = ((b * in_c + ic) * in_h + h_in) * in_w + w_in;
                            int weight_idx = ((ic * out_c + oc) * k + kh) * k + kw;
                            acc += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        // Add bias
        acc += bias[oc];
        
        // Fused Activation Chain Logic
        acc += add_val;
        acc = fminf(acc, 0.0f);
        acc = acc * 0.5f * (1.0f + tanhf(0.79788456f * (acc + 0.044715f * acc * acc * acc)));
        output[tid] = acc * mul_val;
    }
}

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    float add_val,
    float mul_val,
    int stride,
    int padding,
    int output_padding,
    int dilation,
    int groups) {
    
    int batch = input.size(0);
    int in_c = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_c = weight.size(1);
    int k = weight.size(2);
    
    int out_h = (in_h - 1) * stride - 2 * padding + dilation * (k - 1) + output_padding + 1;
    int out_w = (in_w - 1) * stride - 2 * padding + dilation * (k - 1) + output_padding + 1;
    
    int total_elements = batch * out_c * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_conv_transpose_act_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        add_val,
        mul_val,
        batch,
        in_c,
        out_c,
        in_h,
        in_w,
        k,
        stride,
        padding,
        output_padding,
        dilation,
        groups
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    float add_val,
    float mul_val,
    int stride,
    int padding,
    int output_padding,
    int dilation,
    int groups);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op_forward", &fused_op_forward, "Fused ConvTranspose2d + Activation forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose_act',
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
    # Calculate output dimensions
    in_h, in_w = x.shape[2], x.shape[3]
    k = conv_transpose_weight.shape[2]
    out_h = (in_h - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (k - 1) + conv_transpose_output_padding + 1
    out_w = (in_w - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (k - 1) + conv_transpose_output_padding + 1
    out_c = conv_transpose_weight.shape[1]
    
    out = torch.empty((x.shape[0], out_c, out_h, out_w), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op_forward(
        x.contiguous(),
        conv_transpose_weight.contiguous(),
        conv_transpose_bias.contiguous(),
        out,
        add_value,
        multiply_value,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_dilation,
        conv_transpose_groups
    )
    
    return out

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
    return [torch.rand(batch_size, in_channels, height, width)]
