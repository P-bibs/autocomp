# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_094044/code_2.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel code for fused conv transpose 3d + softmax + sigmoid
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Helper function to compute index from 5D coordinates
__device__ inline int get_index_5d(
    int b, int c, int d, int h, int w,
    int channels, int depth, int height, int width
) {
    return b * (channels * depth * height * width) +
           c * (depth * height * width) +
           d * (height * width) +
           h * width +
           w;
}

// Conv transpose 3D kernel
__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    int w_out = idx % out_width;
    int h_out = (idx / out_width) % out_height;
    int d_out = (idx / (out_width * out_height)) % out_depth;
    int c_out = (idx / (out_width * out_height * out_depth)) % out_channels;
    int b = idx / (out_width * out_height * out_depth * out_channels);
    
    float sum = (bias != nullptr) ? bias[c_out] : 0.0f;
    
    // Compute input coordinates that contribute to this output
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int d_in = d_out + padding - kd * dilation;
                int h_in = h_out + padding - kh * dilation;
                int w_in = w_out + padding - kw * dilation;
                
                if (d_in % stride == 0 && h_in % stride == 0 && w_in % stride == 0) {
                    d_in /= stride;
                    h_in /= stride;
                    w_in /= stride;
                    
                    if (d_in >= 0 && d_in < in_depth &&
                        h_in >= 0 && h_in < in_height &&
                        w_in >= 0 && w_in < in_width) {
                        
                        int input_idx = get_index_5d(b, 0, d_in, h_in, w_in, 
                                                    in_channels, in_depth, in_height, in_width);
                        int weight_idx = c_out * (in_channels * kernel_size * kernel_size * kernel_size) +
                                        0 * (kernel_size * kernel_size * kernel_size) +
                                        kd * (kernel_size * kernel_size) +
                                        kh * kernel_size +
                                        kw;
                        
                        for (int c_in = 0; c_in < in_channels; c_in++) {
                            sum += input[input_idx + c_in * (in_depth * in_height * in_width)] *
                                   weight[weight_idx + c_in * (kernel_size * kernel_size * kernel_size)];
                        }
                    }
                }
            }
        }
    }
    
    output[idx] = sum;
}

// Fused softmax + sigmoid kernel
__global__ void fused_softmax_sigmoid_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width,
    int softmax_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * depth * height * width;
    
    if (idx >= total_elements) return;
    
    // Compute strides for accessing input/output
    int w = idx % width;
    int h = (idx / width) % height;
    int d = (idx / (width * height)) % depth;
    int c = (idx / (width * height * depth)) % channels;
    int b = idx / (width * height * depth * channels);
    
    float val = input[idx];
    
    // Handle different softmax dimensions
    float max_val = val;
    float sum_exp = 0.0f;
    
    if (softmax_dim == 1) {
        // Softmax over channel dimension
        // Find max for numerical stability
        for (int ch = 0; ch < channels; ch++) {
            int pidx = get_index_5d(b, ch, d, h, w, channels, depth, height, width);
            max_val = fmaxf(max_val, input[pidx]);
        }
        // Compute sum of exponentials
        for (int ch = 0; ch < channels; ch++) {
            int pidx = get_index_5d(b, ch, d, h, w, channels, depth, height, width);
            sum_exp += expf(input[pidx] - max_val);
        }
    } else if (softmax_dim == 4) {
        // Softmax over width dimension
        for (int ww = 0; ww < width; ww++) {
            int pidx = get_index_5d(b, c, d, h, ww, channels, depth, height, width);
            max_val = fmaxf(max_val, input[pidx]);
        }
        for (int ww = 0; ww < width; ww++) {
            int pidx = get_index_5d(b, c, d, h, ww, channels, depth, height, width);
            sum_exp += expf(input[pidx] - max_val);
        }
    } else {
        // Default: apply softmax on the last dimension (width)
        for (int ww = 0; ww < width; ww++) {
            int pidx = get_index_5d(b, c, d, h, ww, channels, depth, height, width);
            max_val = fmaxf(max_val, input[pidx]);
        }
        for (int ww = 0; ww < width; ww++) {
            int pidx = get_index_5d(b, c, d, h, ww, channels, depth, height, width);
            sum_exp += expf(input[pidx] - max_val);
        }
    }
    
    // Apply softmax
    float softmax_val = (sum_exp > 0.0f) ? expf(val - max_val) / sum_exp : 0.0f;
    
    // Apply sigmoid: sigmoid(x) = 1 / (1 + exp(-x))
    float sigmoid_val = 1.0f / (1.0f + expf(-softmax_val));
    
    output[idx] = sigmoid_val;
}

void fused_conv_transpose3d_softmax_sigmoid_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int output_padding,
    int dilation,
    int softmax_dim
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2); // Assuming cubic kernel
    
    // Calculate output dimensions
    int out_depth = (in_depth - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    
    // Conv transpose 3D
    int conv_total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    int threads_per_block = 256;
    int blocks = (conv_total_elements + threads_per_block - 1) / threads_per_block;
    
    conv_transpose3d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_depth,
        in_height,
        in_width,
        out_depth,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation
    );
    
    cudaDeviceSynchronize();
    
    // Softmax + Sigmoid
    int softmax_total_elements = conv_total_elements;
    blocks = (softmax_total_elements + threads_per_block - 1) / threads_per_block;
    
    fused_softmax_sigmoid_kernel<<<blocks, threads_per_block>>>(
        output.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_channels,
        out_depth,
        out_height,
        out_width,
        softmax_dim
    );
    
    cudaDeviceSynchronize();
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_softmax_sigmoid_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int output_padding,
    int dilation,
    int softmax_dim
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_conv_transpose3d_softmax_sigmoid_forward, 
          "Fused conv transpose 3D, softmax and sigmoid operation");
}
"""

# Compile the extension
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
    softmax_dim,
):
    # Validate groups - our implementation assumes groups=1
    if conv_transpose_groups != 1:
        raise ValueError("This implementation only supports conv_transpose_groups=1")
    
    # Create output tensor with correct shape
    batch_size = x.size(0)
    in_channels = x.size(1)
    in_depth = x.size(2)
    in_height = x.size(3)
    in_width = x.size(4)
    
    out_channels = conv_transpose_weight.size(0)
    kernel_size = conv_transpose_weight.size(2)
    
    out_depth = (in_depth - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + 1 + conv_transpose_output_padding
    out_height = (in_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + 1 + conv_transpose_output_padding
    out_width = (in_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + 1 + conv_transpose_output_padding
    
    output = torch.empty((batch_size, out_channels, out_depth, out_height, out_width), 
                         dtype=x.dtype, device=x.device)
    
    # Call fused operation
    fused_ext.fused_op(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        output,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_dilation,
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
