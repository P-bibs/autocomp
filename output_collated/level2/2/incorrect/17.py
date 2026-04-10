# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_163027/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape', 'scaling_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias', 'scaling_factor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

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
    if 'scaling_factor' in flat_state:
        state_kwargs['scaling_factor'] = flat_state['scaling_factor']
    else:
        state_kwargs['scaling_factor'] = getattr(model, 'scaling_factor')
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

# ----------------------------------------------------------------------
# Fused CUDA kernel: ConvTranspose2d + bias addition + final clamp
# ----------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_1D_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// ConvTranspose2d kernel (simplified version)
__global__ void conv_transpose2d_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding, int output_padding,
    int out_height, int out_width) {
    
    int out_ch = blockIdx.x;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    
    // Shared memory for partial sums
    extern __shared__ float shared_data[];
    
    for (int b = 0; b < batch_size; ++b) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                float sum = 0.0f;
                
                // Compute input coordinates that contribute to this output position
                int start_h = (oh + padding - (kernel_size - 1) + stride - 1) / stride;
                int end_h = min((oh + padding) / stride + 1, in_height);
                int start_w = (ow + padding - (kernel_size - 1) + stride - 1) / stride;
                int end_w = min((ow + padding) / stride + 1, in_width);
                
                start_h = max(0, start_h);
                end_h = min(in_height, end_h);
                start_w = max(0, start_w);
                end_w = min(in_width, end_w);
                
                // Accumulate contributions from all input positions
                for (int ih = start_h; ih < end_h; ++ih) {
                    for (int iw = start_w; iw < end_w; ++iw) {
                        int k_y = oh - stride * ih + padding;
                        int k_x = ow - stride * iw + padding;
                        
                        if (k_y >= 0 && k_y < kernel_size && k_x >= 0 && k_x < kernel_size) {
                            for (int ic = 0; ic < in_channels; ++ic) {
                                float input_val = input[((b * in_channels + ic) * in_height + ih) * in_width + iw];
                                float weight_val = weight[((out_ch * in_channels + ic) * kernel_size + k_y) * kernel_size + k_x];
                                sum += input_val * weight_val;
                            }
                        }
                    }
                }
                
                // Add bias and store result
                int out_idx = ((b * out_channels + out_ch) * out_height + oh) * out_width + ow;
                output[out_idx] = sum + bias[out_ch];
            }
        }
    }
}

// Fused bias addition and clamp kernel
__global__ void fused_bias_clamp_kernel(const float* __restrict__ input,
                                         const float* __restrict__ bias,
                                         float* __restrict__ output,
                                         int N, int C, int H, int W,
                                         float max_clamp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx < total) {
        int n = idx / (C * H * W);
        int rem = idx % (C * H * W);
        int c = rem / (H * W);
        float val = input[idx] + bias[c];
        if (val < 0.0f) val = 0.0f;
        else if (val > max_clamp) val = max_clamp;
        output[idx] = val;
    }
}

void run_conv_transpose2d(const at::Tensor& input,
                          const at::Tensor& weight,
                          const at::Tensor& bias,
                          at::Tensor& output,
                          int stride, int padding, int output_padding) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    // Launch conv transpose kernel
    dim3 grid(out_channels);
    dim3 block(256);
    size_t shared_mem_size = block.x * sizeof(float);
    
    conv_transpose2d_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_size, stride, padding, output_padding,
        out_height, out_width
    );
    
    cudaDeviceSynchronize();
}

void run_fused_bias_clamp(const at::Tensor& input,
                          const at::Tensor& bias,
                          at::Tensor& output,
                          float max_clamp) {
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    const int total = N * C * H * W;
    const int threads = 1024;
    const int blocks  = (total + threads - 1) / threads;
    
    fused_bias_clamp_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, max_clamp);
    
    cudaDeviceSynchronize();
}
"""

cpp_source = r"""
#include <torch/extension.h>

void run_conv_transpose2d(const at::Tensor& input,
                          const at::Tensor& weight,
                          const at::Tensor& bias,
                          at::Tensor& output,
                          int stride, int padding, int output_padding);

void run_fused_bias_clamp(const at::Tensor& input,
                          const at::Tensor& bias,
                          at::Tensor& output,
                          float max_clamp);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_conv_transpose2d", &run_conv_transpose2d, "Custom ConvTranspose2d");
    m.def("run_fused_bias_clamp", &run_fused_bias_clamp, "Fused bias add + clamp");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
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
    bias,
    scaling_factor,
):
    # Assuming groups=1 and dilation=1 for simplification as per original parameters
    # Calculate output dimensions
    in_height, in_width = x.shape[2], x.shape[3]
    kernel_size = conv_transpose_weight.shape[2]
    out_height = (in_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    out_width = (in_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    
    # Create temporary output for conv transpose
    conv_out = torch.empty(x.shape[0], conv_transpose_weight.shape[0], out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Run custom conv transpose
    fused_ext.run_conv_transpose2d(
        x, conv_transpose_weight, conv_transpose_bias, conv_out,
        conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding
    )
    
    # Fuse bias addition + the clamp chain
    max_clamp = 1.0 / scaling_factor
    N, C, H, W = conv_out.shape
    bias_1d = bias.view(-1)  # (C,) – bias has shape (C,1,1)
    out = torch.empty_like(conv_out)  # output buffer on the same device
    
    fused_ext.run_fused_bias_clamp(conv_out, bias_1d, out, max_clamp)
    return out

# ----------------------------------------------------------------------
# Helper functions required by the harness
# ----------------------------------------------------------------------
batch_size = 128
in_channels  = 64
out_channels = 64
height = width = 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding,
            output_padding, bias_shape, scaling_factor]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
