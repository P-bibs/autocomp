# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_123254/code_11.py
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- CUDA Kernel for Fused ConvTranspose2d + Bias Sub + Tanh ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

// Helper to get valid indices
__device__ __forceinline__ bool within_bounds(int h, int w, int H, int W) {
    return (h >= 0 && h < H && w >= 0 && w < W);
}

__global__ void fused_conv_transpose_bias_tanh_kernel(
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> bias,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> output,
    int in_channels, int out_channels,
    int input_height, int input_width,
    int kernel_size, int stride, int padding, int output_padding,
    int dilation, int groups) {

    int batch_idx = blockIdx.x;
    int out_ch = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    if (batch_idx >= input.size(0) || out_ch >= out_channels) return;

    int output_height = (input_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;

    // Shared memory for partial sums
    extern __shared__ float sdata[];
    float* shared_output = sdata;

    for (int out_y = tid; out_y < output_height; out_y += total_threads) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
            float sum = 0.0f;

            // ConvTranspose logic
            for (int k_c = 0; k_c < in_channels / groups; ++k_c) {
                int group_id = out_ch / (out_channels / groups);
                int in_ch = group_id * (in_channels / groups) + k_c;

                for (int k_h = 0; k_h < kernel_size; ++k_h) {
                    for (int k_w = 0; k_w < kernel_size; ++k_w) {
                        int in_y_origin = out_y + padding - k_h * dilation;
                        int in_x_origin = out_x + padding - k_w * dilation;

                        if (in_y_origin % stride == 0 && in_x_origin % stride == 0) {
                            int in_y = in_y_origin / stride;
                            int in_x = in_x_origin / stride;

                            if (within_bounds(in_y, in_x, input_height, input_width)) {
                                float val = input[batch_idx][in_ch][in_y][in_x];
                                float wgt = weight[out_ch][k_c][k_h][k_w];
                                sum += val * wgt;
                            }
                        }
                    }
                }
            }

            // Subtract bias and apply tanh
            float biased_val = sum - bias[out_ch];
            float result = tanhf(biased_val);

            // Write result
            output[batch_idx][out_ch][out_y][out_x] = result;
        }
    }
}

void launch_fused_conv_transpose_bias_tanh(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    int kernel_size, int stride, int padding, int output_padding,
    int dilation, int groups) {

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);
    auto out_channels = weight.size(0);

    dim3 block(32, 32);
    dim3 grid(batch_size, (out_channels + block.y - 1) / block.y);

    int shared_mem_size = 0;

    fused_conv_transpose_bias_tanh_kernel<<<grid, block, shared_mem_size>>>(
        input.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        weight.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        bias.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        output.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        in_channels, out_channels,
        input_height, input_width,
        kernel_size, stride, padding, output_padding,
        dilation, groups
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void launch_fused_conv_transpose_bias_tanh(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    int kernel_size, int stride, int padding, int output_padding,
    int dilation, int groups);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_bias_tanh", &launch_fused_conv_transpose_bias_tanh, "Fused ConvTranspose2d + Bias Sub + Tanh");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose_bias_tanh_ext',
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
    # Allocate output tensor with correct shape
    batch_size, in_channels, input_height, input_width = x.shape
    out_channels, _, kernel_h, kernel_w = conv_transpose_weight.shape
    assert kernel_h == kernel_w, "Only square kernels supported"
    kernel_size = kernel_h

    output_height = (input_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + 1 + conv_transpose_output_padding
    output_width = (input_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + 1 + conv_transpose_output_padding
    
    output = torch.empty((batch_size, out_channels, output_height, output_width), device=x.device, dtype=x.dtype)
    
    # Call the custom fused kernel
    fused_ext.fused_conv_transpose_bias_tanh(
        x, conv_transpose_weight, conv_transpose_bias.flatten(), output,
        kernel_size, conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding,
        conv_transpose_dilation, conv_transpose_groups
    )
    
    return output

batch_size = 32
in_channels  = 64  
out_channels = 64  
height = width = 256 
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

