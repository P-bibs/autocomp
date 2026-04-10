# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_153448/code_2.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// Vectorized GELU
__device__ __forceinline__ float4 fast_gelu_vec(float4 v) {
    auto gelu = [](float x) {
        return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    };
    return make_float4(gelu(v.x), gelu(v.y), gelu(v.z), gelu(v.w));
}

// Vectorized fused operation kernel
__global__ void fused_op_vectorized_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                           float add_val, float mul_val, int num_elements) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < num_elements) {
        float4 in_vec = reinterpret_cast<const float4*>(input)[idx / 4];
        
        in_vec.x = fminf(in_vec.x + add_val, 0.0f);
        in_vec.y = fminf(in_vec.y + add_val, 0.0f);
        in_vec.z = fminf(in_vec.z + add_val, 0.0f);
        in_vec.w = fminf(in_vec.w + add_val, 0.0f);
        
        in_vec = fast_gelu_vec(in_vec);
        
        in_vec.x *= mul_val; in_vec.y *= mul_val;
        in_vec.z *= mul_val; in_vec.w *= mul_val;
        
        reinterpret_cast<float4*>(output)[idx / 4] = in_vec;
    }
}

// Conv2d transpose kernel (Rule #6 implementation)
__global__ void conv_transpose2d_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding,
    int output_padding, int groups, int dilation,
    int out_height, int out_width) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * out_height * out_width;
    if (tid >= total_output_elements) return;

    int tmp = tid;
    int w_out = tmp % out_width; tmp /= out_width;
    int h_out = tmp % out_height; tmp /= out_height;
    int c_out = tmp % out_channels; tmp /= out_channels;
    int n = tmp;

    float value = (bias != nullptr) ? bias[c_out] : 0.0f;
    
    int c_per_group = in_channels / groups;
    int g = c_out / (out_channels / groups);
    int start_c_in = g * c_per_group;
    int end_c_in = start_c_in + c_per_group;

    for (int c_in = start_c_in; c_in < end_c_in; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h_out + padding - kh * dilation;
                int w_in = w_out + padding - kw * dilation;
                if (h_in % stride == 0 && w_in % stride == 0) {
                    h_in /= stride;
                    w_in /= stride;
                    if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                        int input_idx = ((n * in_channels + c_in) * in_height + h_in) * in_width + w_in;
                        int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw;
                        value += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    output[tid] = value;
}

void launch_fused_op(torch::Tensor input, torch::Tensor output, float add_val, float mul_val) {
    int num_elements = input.numel();
    int threads = 128;
    int blocks = (num_elements / 4 + threads - 1) / threads;
    fused_op_vectorized_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), add_val, mul_val, num_elements);
}

void launch_conv_transpose2d(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output,
    int kernel_size, int stride, int padding,
    int output_padding, int groups, int dilation) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(1);
    
    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    
    int total_output_elements = batch_size * out_channels * out_height * out_width;
    int threads = 256;
    int blocks = (total_output_elements + threads - 1) / threads;
    
    const float* bias_ptr = (bias.numel() > 0) ? bias.data_ptr<float>() : nullptr;
    
    conv_transpose2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
        output.data_ptr<float>(),
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_size, stride, padding,
        output_padding, groups, dilation,
        out_height, out_width
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_fused_op(torch::Tensor input, torch::Tensor output, float add_val, float mul_val);
void launch_conv_transpose2d(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output,
    int kernel_size, int stride, int padding,
    int output_padding, int groups, int dilation);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_fused_op, "Vectorized fused operation");
    m.def("conv_transpose2d", &launch_conv_transpose2d, "Custom ConvTranspose2D");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_custom_ops',
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
    # Custom ConvTranspose2d implementation without PyTorch built-in
    out_h = (x.size(2) - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + conv_transpose_dilation[0] * (conv_transpose_weight.size(2) - 1) + conv_transpose_output_padding[0] + 1
    out_w = (x.size(3) - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + conv_transpose_dilation[1] * (conv_transpose_weight.size(3) - 1) + conv_transpose_output_padding[1] + 1
    conv_out = torch.empty((x.size(0), conv_transpose_weight.size(1), out_h, out_w), device=x.device, dtype=x.dtype)
    
    fused_ext.conv_transpose2d(
        x.contiguous(), 
        conv_transpose_weight.contiguous(), 
        conv_transpose_bias.contiguous() if conv_transpose_bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        conv_out,
        conv_transpose_weight.size(2),  # kernel_size assumed square
        conv_transpose_stride[0],       # stride assumed uniform
        conv_transpose_padding[0],      # padding assumed uniform
        conv_transpose_output_padding[0], # output_padding assumed uniform
        conv_transpose_groups,
        conv_transpose_dilation[0]      # dilation assumed uniform
    )
    
    # Fused vectorized operation
    out = torch.empty_like(conv_out)
    fused_ext.fused_op(conv_out.contiguous(), out, float(add_value), float(multiply_value))
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
