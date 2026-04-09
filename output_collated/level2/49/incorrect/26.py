# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_094958/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'softmax_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, applies Softmax and Sigmoid.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

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
    # State for softmax (nn.Softmax)
    state_kwargs['softmax_dim'] = model.softmax.dim
    # State for sigmoid (nn.Sigmoid)
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

# Optimization: Replace PyTorch's conv_transpose3d with a custom CUDA kernel and fuse softmax + sigmoid operations
# This provides full end-to-end optimization by eliminating all intermediate memory buffers.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

// 3D Transposed Convolution + Softmax + Sigmoid fusion kernel
__global__ void fused_conv_transpose3d_softmax_sigmoid_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    int softmax_dim
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    
    if (idx >= total_elements) return;
    
    // Decompose linear index into multi-dimensional indices
    int w_out = idx % output_w;
    int h_out = (idx / output_w) % output_h;
    int d_out = (idx / (output_w * output_h)) % output_d;
    int c_out = (idx / (output_w * output_h * output_d)) % out_channels;
    int batch = idx / (output_w * output_h * output_d * out_channels);
    
    // Initialize output with bias
    float value = bias[c_out];
    
    // Convolution transpose calculation
    // For each input channel in the group
    int group_idx = c_out / (out_channels / groups);
    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;
    
    for (int c_in_group = 0; c_in_group < in_channels_per_group; ++c_in_group) {
        int c_in = group_idx * in_channels_per_group + c_in_group;
        
        // Iterate through kernel elements
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    // Calculate corresponding input position
                    int d_in = d_out - kd * dilation + 2 * padding;
                    int h_in = h_out - kh * dilation + 2 * padding;
                    int w_in = w_out - kw * dilation + 2 * padding;
                    
                    // Check if divisible by stride
                    if (d_in % stride == 0 && h_in % stride == 0 && w_in % stride == 0) {
                        d_in /= stride;
                        h_in /= stride;
                        w_in /= stride;
                        
                        // Check input bounds
                        if (d_in >= 0 && d_in < input_d &&
                            h_in >= 0 && h_in < input_h &&
                            w_in >= 0 && w_in < input_w) {
                            
                            // Calculate indices
                            int input_idx = batch * (in_channels * input_d * input_h * input_w) +
                                          c_in * (input_d * input_h * input_w) +
                                          d_in * (input_h * input_w) +
                                          h_in * input_w +
                                          w_in;
                                          
                            int weight_idx = c_in * (out_channels_per_group * kernel_size * kernel_size * kernel_size) +
                                           (c_out % out_channels_per_group) * (kernel_size * kernel_size * kernel_size) +
                                           kd * (kernel_size * kernel_size) +
                                           kh * kernel_size +
                                           kw;
                                           
                            value += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Store the conv result temporarily
    output[idx] = value;
}

__global__ void fused_softmax_sigmoid_kernel(float* data, int N, int C, int D, int H, int W, int softmax_dim) {
    if (softmax_dim == 1) { // Channel dimension
        int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int batch_idx = blockIdx.y;
        
        int spatial_size = D * H * W;
        if (spatial_idx >= spatial_size) return;
        
        int d = spatial_idx / (H * W);
        int h = (spatial_idx / W) % H;
        int w = spatial_idx % W;
        
        // Find max for numerical stability
        float max_val = -1e20f;
        for (int c = 0; c < C; ++c) {
            int idx = batch_idx * (C * D * H * W) + c * (D * H * W) + d * (H * W) + h * W + w;
            max_val = fmaxf(max_val, data[idx]);
        }

        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            int idx = batch_idx * (C * D * H * W) + c * (D * H * W) + d * (H * W) + h * W + w;
            float val = expf(data[idx] - max_val);
            data[idx] = val;
            sum += val;
        }

        // Apply Softmax and Sigmoid
        for (int c = 0; c < C; ++c) {
            int idx = batch_idx * (C * D * H * W) + c * (D * H * W) + d * (H * W) + h * W + w;
            float softmax_out = data[idx] / sum;
            data[idx] = 1.0f / (1.0f + expf(-softmax_out));
        }
    }
}

void fused_model_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    int softmax_dim
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_d = input.size(2);
    int input_h = input.size(3);
    int input_w = input.size(4);
    
    int out_channels = weight.size(1);
    
    int output_d = (input_d - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int output_h = (input_h - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int output_w = (input_w - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    
    // Launch conv transpose kernel
    int conv_threads = 256;
    int conv_elements = batch_size * out_channels * output_d * output_h * output_w;
    int conv_blocks = (conv_elements + conv_threads - 1) / conv_threads;
    
    fused_conv_transpose3d_softmax_sigmoid_kernel<<<conv_blocks, conv_threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        softmax_dim
    );
    
    // Launch fused softmax + sigmoid kernel
    if (softmax_dim == 1) {
        int spatial_size = output_d * output_h * output_w;
        int softmax_threads = 256;
        dim3 softmax_blocks((spatial_size + softmax_threads - 1) / softmax_threads, batch_size);
        
        fused_softmax_sigmoid_kernel<<<softmax_blocks, softmax_threads>>>(
            output.data_ptr<float>(),
            batch_size,
            out_channels,
            output_d,
            output_h,
            output_w,
            softmax_dim
        );
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_model_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    int softmax_dim
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_model_forward", &fused_model_forward, "Fused Conv Transpose 3D + Softmax + Sigmoid forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_model',
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
    softmax_dim,
):
    # Calculate output dimensions
    batch_size = x.size(0)
    in_channels = x.size(1)
    input_d, input_h, input_w = x.size(2), x.size(3), x.size(4)
    
    out_channels = conv_transpose_weight.size(1)
    kernel_size = conv_transpose_weight.size(2)
    
    stride = conv_transpose_stride[0] if isinstance(conv_transpose_stride, (list, tuple)) else conv_transpose_stride
    padding = conv_transpose_padding[0] if isinstance(conv_transpose_padding, (list, tuple)) else conv_transpose_padding
    output_padding = conv_transpose_output_padding[0] if isinstance(conv_transpose_output_padding, (list, tuple)) else conv_transpose_output_padding
    dilation = conv_transpose_dilation[0] if isinstance(conv_transpose_dilation, (list, tuple)) else conv_transpose_dilation
    
    # Calculate output dimensions
    output_d = (input_d - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    output_h = (input_h - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    output_w = (input_w - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_d, output_h, output_w, device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_model_forward(
        x.contiguous(),
        conv_transpose_weight.contiguous(),
        conv_transpose_bias.contiguous(),
        output,
        kernel_size,
        stride,
        padding,
        output_padding,
        conv_transpose_groups,
        dilation,
        softmax_dim
    )
    
    return output

batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]
