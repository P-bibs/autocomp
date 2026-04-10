# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_152515/code_0.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t gelu_impl(scalar_t x) {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    const scalar_t c1 = 0.7978845608028654;  // sqrt(2/pi)
    const scalar_t c2 = 0.044715;
    scalar_t x3 = x * x * x;
    scalar_t arg = c1 * (x + c2 * x3);
    return 0.5 * x * (1.0 + tanh(arg));
}

template <typename scalar_t>
__global__ void fused_conv_transpose2d_elementwise_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ add_value,
    const scalar_t* __restrict__ multiply_value,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_padding_h,
    int output_padding_w,
    int groups,
    int dilation_h,
    int dilation_w) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (tid >= total_elements) return;
    
    int w = tid % out_width;
    int h = (tid / out_width) % out_height;
    int c = (tid / (out_width * out_height)) % out_channels;
    int b = tid / (out_width * out_height * out_channels);
    
    scalar_t sum = (bias != nullptr) ? bias[c] : 0.0;
    
    // Conv transpose calculation
    int group_idx = c / (out_channels / groups);
    int channels_per_group = in_channels / groups;
    
    for (int i = 0; i < kernel_h; ++i) {
        for (int j = 0; j < kernel_w; ++j) {
            // Calculate corresponding input position
            int in_h = h + padding_h - i * dilation_h;
            int in_w = w + padding_w - j * dilation_w;
            
            if (in_h % stride_h == 0 && in_w % stride_w == 0) {
                in_h /= stride_h;
                in_w /= stride_w;
                
                if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                    for (int ic = 0; ic < channels_per_group; ++ic) {
                        int input_channel = group_idx * channels_per_group + ic;
                        int input_idx = ((b * in_channels + input_channel) * in_height + in_h) * in_width + in_w;
                        int weight_idx = (ic * out_channels + c) * kernel_h * kernel_w + i * kernel_w + j;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Element-wise operations
    sum += *add_value;
    sum = fminf(sum, 0.0f);
    sum = gelu_impl(sum);
    sum *= *multiply_value;
    
    output[tid] = sum;
}

void fused_conv_transpose2d_elementwise_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& add_val,
    const at::Tensor& multiply_val,
    at::Tensor& output,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t output_padding_h,
    int64_t output_padding_w,
    int64_t groups,
    int64_t dilation_h,
    int64_t dilation_w) {
    
    const at::cuda::CUDAGuard device_guard(input.device());
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);
    
    auto out_height = (in_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + 1 + output_padding_h;
    auto out_width = (in_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + 1 + output_padding_w;
    
    auto total_elements = batch_size * out_channels * out_height * out_width;
    const int threads = 512;
    const int blocks = (total_elements + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_conv_transpose2d_elementwise_forward", ([&] {
        const scalar_t* input_ptr = input.data_ptr<scalar_t>();
        const scalar_t* weight_ptr = weight.data_ptr<scalar_t>();
        const scalar_t* bias_ptr = bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr;
        const scalar_t* add_val_ptr = add_val.data_ptr<scalar_t>();
        const scalar_t* multiply_val_ptr = multiply_val.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();
        
        fused_conv_transpose2d_elementwise_kernel<scalar_t><<<blocks, threads>>>(
            input_ptr,
            weight_ptr,
            bias_ptr,
            add_val_ptr,
            multiply_val_ptr,
            output_ptr,
            batch_size,
            in_channels,
            out_channels,
            in_height,
            in_width,
            out_height,
            out_width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            output_padding_h,
            output_padding_w,
            groups,
            dilation_h,
            dilation_w
        );
    }));
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose2d_elementwise_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& add_val,
    const at::Tensor& multiply_val,
    at::Tensor& output,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t output_padding_h,
    int64_t output_padding_w,
    int64_t groups,
    int64_t dilation_h,
    int64_t dilation_w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_conv_transpose2d_elementwise_forward, "Fused ConvTranspose2d with elementwise operations");
}
"""

fused_ext = load_inline(
    name='fused_op',
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
    # Handle both single value and tuple inputs for stride, padding, etc.
    if isinstance(conv_transpose_stride, int):
        stride_h = stride_w = conv_transpose_stride
    else:
        stride_h, stride_w = conv_transpose_stride
    
    if isinstance(conv_transpose_padding, int):
        padding_h = padding_w = conv_transpose_padding
    else:
        padding_h, padding_w = conv_transpose_padding
        
    if isinstance(conv_transpose_output_padding, int):
        output_padding_h = output_padding_w = conv_transpose_output_padding
    else:
        output_padding_h, output_padding_w = conv_transpose_output_padding
        
    if isinstance(conv_transpose_dilation, int):
        dilation_h = dilation_w = conv_transpose_dilation
    else:
        dilation_h, dilation_w = conv_transpose_dilation

    batch_size = x.size(0)
    in_channels = x.size(1)
    in_height = x.size(2)
    in_width = x.size(3)
    out_channels = conv_transpose_weight.size(0)
    
    kernel_h = conv_transpose_weight.size(2)
    kernel_w = conv_transpose_weight.size(3)
    
    out_height = (in_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + 1 + output_padding_h
    out_width = (in_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + 1 + output_padding_w
    
    output = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    add_tensor = torch.tensor([add_value], device=x.device, dtype=x.dtype)
    multiply_tensor = torch.tensor([multiply_value], device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        add_tensor,
        multiply_tensor,
        output,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
        conv_transpose_groups,
        dilation_h,
        dilation_w
    )
    
    return output

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
