# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_140117/code_8.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, followed by a sum, 
    a residual add, a multiplication, and another residual add.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
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

# Optimized CUDA kernel fusing conv_transpose3d and post-processing
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

__global__ void fused_conv_transpose3d_post_process_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ post_bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    extern __shared__ float s_post_bias[];
    
    // Load post-processing bias into shared memory
    for (int i = threadIdx.x; i < out_channels; i += blockDim.x) {
        s_post_bias[i] = post_bias[i];
    }
    __syncthreads();
    
    // Calculate global thread indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (idx >= total_output_elements) return;
    
    // Decode output tensor indices
    int tmp = idx;
    int w_out = tmp % output_width;
    tmp /= output_width;
    int h_out = tmp % output_height;
    tmp /= output_height;
    int d_out = tmp % output_depth;
    tmp /= output_depth;
    int c_out = tmp % out_channels;
    int n = tmp / out_channels;
    
    // Load post-processing bias for this channel
    float b = s_post_bias[c_out];
    
    // Compute convolution value at this output position
    float conv_value = 0.0f;
    
    // Loop over kernel positions and input channels
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                // Map output position to input position
                int d_in = d_out + padding - kd;
                int h_in = h_out + padding - kh;
                int w_in = w_out + padding - kw;
                
                // Check if valid input position
                if (d_in % stride == 0 && h_in % stride == 0 && w_in % stride == 0) {
                    d_in /= stride;
                    h_in /= stride;
                    w_in /= stride;
                    
                    if (d_in >= 0 && d_in < input_depth &&
                        h_in >= 0 && h_in < input_height &&
                        w_in >= 0 && w_in < input_width) {
                        
                        // Loop over input channels (assuming groups=1)
                        for (int c_in = 0; c_in < in_channels; ++c_in) {
                            // Calculate indices
                            int input_idx = ((((n * in_channels) + c_in) * input_depth + d_in) * input_height + h_in) * input_width + w_in;
                            int weight_idx = ((((c_out * in_channels) + c_in) * kernel_size + kd) * kernel_size + kh) * kernel_size + kw;
                            
                            // Accumulate convolution
                            conv_value += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add bias
    conv_value += conv_bias[c_out];
    
    // Apply post-processing: ((x + b) + x) * x + x = (2x + b) * x + x
    float result = ((conv_value + b) + conv_value) * conv_value + conv_value;
    
    // Write output
    output[idx] = result;
}

void fused_conv_transpose3d_post_process_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    // Get tensor dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int out_channels = weight.size(1); // For conv_transpose: weight is (in_channels, out_channels/groups, kD, kH, kW)
    
    // Calculate output dimensions
    int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    // Total number of output elements
    int64_t total_output_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    // Configure kernel launch parameters
    int threads_per_block = min(MAX_THREADS_PER_BLOCK, (int)total_output_elements);
    int blocks = (total_output_elements + threads_per_block - 1) / threads_per_block;
    
    // Calculate shared memory size for post-processing bias
    size_t shared_mem_size = out_channels * sizeof(float);
    
    // Launch kernel
    fused_conv_transpose3d_post_process_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        post_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        output_padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_post_process_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
);

torch::Tensor fused_conv_transpose3d_post_process(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    // Calculate output dimensions
    int64_t output_depth = (input.size(2) - 1) * stride - 2 * padding + kernel_size + output_padding;
    int64_t output_height = (input.size(3) - 1) * stride - 2 * padding + kernel_size + output_padding;
    int64_t output_width = (input.size(4) - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output_tensor = torch::empty(
        {input.size(0), weight.size(1), output_depth, output_height, output_width},
        input.options()
    );
    
    fused_conv_transpose3d_post_process_forward(
        input,
        weight,
        conv_bias,
        post_bias,
        output_tensor,
        kernel_size,
        stride,
        padding,
        output_padding
    );
    
    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_post_process", &fused_conv_transpose3d_post_process, 
          "Fused 3D transposed convolution and post-processing");
}
"""

fused_ext = load_inline(
    name='fused_conv_transpose3d_post_process_ext',
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
    # Ensure inputs are on GPU
    x = x.cuda()
    conv_transpose_weight = conv_transpose_weight.cuda()
    conv_transpose_bias = conv_transpose_bias.cuda()
    bias = bias.cuda()
    
    # Verify groups is 1 (assumption in our kernel)
    if conv_transpose_groups != 1:
        raise ValueError("Only groups=1 is supported")
    
    # Verify dilation is (1,1,1) (assumption in our kernel)
    if conv_transpose_dilation != (1, 1, 1):
        raise ValueError("Only dilation=(1,1,1) is supported")
    
    # Extract kernel size (assuming cubic kernel)
    kernel_size = conv_transpose_weight.size(2)
    
    # Flatten bias for simplified kernel indexing
    bias_flat = bias.view(-1)
    
    # Use optimized fused kernel for both conv_transpose3d and post-processing
    return fused_ext.fused_conv_transpose3d_post_process(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        bias_flat,
        kernel_size,
        conv_transpose_stride[0],  # Assuming uniform stride
        conv_transpose_padding[0],  # Assuming uniform padding
        conv_transpose_output_padding[0]  # Assuming uniform output padding
    )

# Model parameters
batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
