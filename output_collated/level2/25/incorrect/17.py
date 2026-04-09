# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_083021/code_5.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

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
    # State for conv (nn.Conv2d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
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

# --- CUDA Kernel Code ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAMathCompat.h> // for compat::tanh

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // Calculate output dimensions
    int out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * out_channels * out_h * out_w;

    if (tid >= total_threads) return;

    // Decompose linear index
    int w_out = tid % out_w;
    int h_out = (tid / out_w) % out_h;
    int c_out = (tid / (out_w * out_h)) % out_channels;
    int n = tid / (out_w * out_h * out_channels);

    float sum = 0.0f;
    int h_in_start = h_out * stride - padding;
    int w_in_start = w_out * stride - padding;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h_in_start + kh * dilation;
                int w_in = w_in_start + kw * dilation;

                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int input_idx = ((n * in_channels + c_in) * height + h_in) * width + w_in;
                    int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    sum += bias[c_out];
    output[tid] = sum;
}

__global__ void fused_min_tanh_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_h,
    int out_w
) {
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial = batch_size * out_h * out_w;
    
    if (spatial_idx >= total_spatial) return;
    
    int b = spatial_idx / (out_h * out_w);
    int hw = spatial_idx % (out_h * out_w);
    int h = hw / out_w;
    int w = hw % out_w;
    
    // Min reduction along channel dimension
    float min_val = input[((b * in_channels) * out_h + h) * out_w + w];
    for (int c = 1; c < in_channels; ++c) {
        float val = input[((b * in_channels + c) * out_h + h) * out_w + w];
        if (val < min_val) min_val = val;
    }
    
    // Apply tanh twice (equivalent to one tanh)
    float result = ::tanhf(min_val);
    result = ::tanhf(result);
    
    output[spatial_idx] = result;
}

void fused_model_forward(
    const torch::Tensor input,
    const torch::Tensor conv_weight,
    const torch::Tensor conv_bias,
    torch::Tensor output_conv,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = conv_weight.size(0);
    const int kernel_size = conv_weight.size(2);
    
    const int out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Launch conv2d kernel
    const int conv_total_threads = batch_size * out_channels * out_h * out_w;
    const int conv_block_size = 256;
    const int conv_grid_size = (conv_total_threads + conv_block_size - 1) / conv_block_size;
    
    conv2d_kernel<<<conv_grid_size, conv_block_size>>>(
        input.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        output_conv.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        dilation
    );
    
    // Launch fused min+tanh kernel
    const int fused_total_threads = batch_size * out_h * out_w;
    const int fused_block_size = 256;
    const int fused_grid_size = (fused_total_threads + fused_block_size - 1) / fused_block_size;
    
    fused_min_tanh_kernel<<<fused_grid_size, fused_block_size>>>(
        output_conv.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_channels,
        out_h,
        out_w
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_model_forward(
    const torch::Tensor input,
    const torch::Tensor conv_weight,
    const torch::Tensor conv_bias,
    torch::Tensor output_conv,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_model", &fused_model_forward, "Fused model forward pass");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_model_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
):
    # Validate that conv_groups is 1 (our kernel doesn't support groups)
    if conv_groups != 1:
        raise ValueError("conv_groups must be 1 for custom kernel")
    
    batch_size = x.size(0)
    in_channels = x.size(1)
    height = x.size(2)
    width = x.size(3)
    out_channels = conv_weight.size(0)
    kernel_size = conv_weight.size(2)
    
    # Calculate output dimensions
    out_h = (height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_w = (width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    # Allocate intermediate and output tensors
    output_conv = torch.empty((batch_size, out_channels, out_h, out_w), device=x.device, dtype=x.dtype)
    output = torch.empty((batch_size, 1, out_h, out_w), device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_model(
        x, conv_weight, conv_bias, output_conv, output,
        conv_stride, conv_padding, conv_dilation
    )
    
    return output

batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
