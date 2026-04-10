# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_155151/code_2.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Fast GELU approximation
__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Fused kernel with grid-stride loop for element-wise operations
__global__ void fused_op_kernel(const float* input, float* output, float add_val, float mul_val, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < num_elements; i += stride) {
        float val = input[i] + add_val;
        val = fminf(val, 0.0f);
        val = fast_gelu(val);
        output[i] = val * mul_val;
    }
}

// Conv transpose 2D kernel using implicit GEMM
__global__ void conv_transpose2d_implicit_gemm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int out_pad_h,
    int out_pad_w,
    int dilation_h,
    int dilation_w,
    int out_height,
    int out_width
) {
    // Calculate global thread index
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * out_height * out_width;
    
    if (out_idx >= total_output_elements) return;
    
    // Decode output index
    int tmp = out_idx;
    int w_out = tmp % out_width; tmp /= out_width;
    int h_out = tmp % out_height; tmp /= out_height;
    int c_out = tmp % out_channels; tmp /= out_channels;
    int n = tmp;
    
    float sum = (bias != nullptr) ? bias[c_out] : 0.0f;
    
    // Compute input region that contributes to this output element
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            // Calculate corresponding input position
            int h_in = h_out - stride_h * kh + pad_h;
            int w_in = w_out - stride_w * kw + pad_w;
            
            // Check bounds and add contribution
            if (h_in >= 0 && h_in < in_height * stride_h && 
                w_in >= 0 && w_in < in_width * stride_w &&
                h_in % stride_h == 0 && w_in % stride_w == 0) {
                
                int src_h = h_in / stride_h;
                int src_w = w_in / stride_w;
                
                if (src_h < in_height && src_w < in_width) {
                    int input_idx = ((n * in_channels + c_out) * in_height + src_h) * in_width + src_w;
                    int weight_idx = (c_out * kernel_h + kh) * kernel_w + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    output[out_idx] = sum;
}

// Wrapper functions
void fused_op_forward(torch::Tensor input, torch::Tensor output, float add_val, float mul_val) {
    int num_elements = input.numel();
    int threads = 256;
    int blocks = min((num_elements + threads - 1) / threads, 65535);
    fused_op_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), add_val, mul_val, num_elements);
}

void conv_transpose2d_implicit_gemm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int out_pad_h,
    int out_pad_w,
    int dilation_h,
    int dilation_w
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_channels = weight.size(1);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    
    int out_height = (in_height - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + out_pad_h + 1;
    int out_width = (in_width - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + out_pad_w + 1;
    
    int total_output_elements = batch_size * out_channels * out_height * out_width;
    int threads = 256;
    int blocks = min((total_output_elements + threads - 1) / threads, 65535);
    
    conv_transpose2d_implicit_gemm_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        out_pad_h,
        out_pad_w,
        dilation_h,
        dilation_w,
        out_height,
        out_width
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor input, torch::Tensor output, float add_val, float mul_val);
void conv_transpose2d_implicit_gemm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int out_pad_h,
    int out_pad_w,
    int dilation_h,
    int dilation_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused op with grid-stride loop");
    m.def("conv_transpose2d_implicit_gemm", &conv_transpose2d_implicit_gemm_forward, "Conv transpose 2D with implicit GEMM");
}
"""

# Compile the extension
custom_ext = load_inline(
    name='custom_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
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
    # Since groups > 1 is not handled in our custom kernel, fallback to PyTorch for that case
    # But according to rule #6, we should not use built-in functions.
    # For the purpose of this optimization, we assume groups=1 (common case)
    if conv_transpose_groups != 1:
        raise NotImplementedError("Custom conv transpose implementation only supports groups=1")
        
    # Allocate output tensor for conv transpose
    out_channels = conv_transpose_weight.size(1)
    kernel_h, kernel_w = conv_transpose_weight.size(2), conv_transpose_weight.size(3)
    stride_h, stride_w = conv_transpose_stride if isinstance(conv_transpose_stride, (tuple, list)) else (conv_transpose_stride, conv_transpose_stride)
    pad_h, pad_w = conv_transpose_padding if isinstance(conv_transpose_padding, (tuple, list)) else (conv_transpose_padding, conv_transpose_padding)
    out_pad_h, out_pad_w = conv_transpose_output_padding if isinstance(conv_transpose_output_padding, (tuple, list)) else (conv_transpose_output_padding, conv_transpose_output_padding)
    dilation_h, dilation_w = conv_transpose_dilation if isinstance(conv_transpose_dilation, (tuple, list)) else (conv_transpose_dilation, conv_transpose_dilation)
    
    batch_size, in_channels, in_height, in_width = x.shape
    
    out_height = (in_height - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + out_pad_h + 1
    out_width = (in_width - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + out_pad_w + 1
    
    conv_out = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)
    
    # Perform custom conv transpose
    custom_ext.conv_transpose2d_implicit_gemm(
        x, conv_transpose_weight, conv_transpose_bias, conv_out,
        stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w, dilation_h, dilation_w
    )
    
    # Apply fused operations with grid-stride loop
    final_out = torch.empty_like(conv_out)
    custom_ext.fused_op(conv_out, final_out, float(add_value), float(multiply_value))
    
    return final_out

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
