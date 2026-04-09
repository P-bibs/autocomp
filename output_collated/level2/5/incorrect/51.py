# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_121229/code_8.py
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
from torch.utils.cpp_extension import load_inline

# CUDA kernel to perform fused bias subtraction and tanh activation with vectorization
# The kernel processes (N, C, H, W) where bias is (C, 1, 1).
# We use grid-stride loop with float4 vectorization for better memory bandwidth utilization.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bias_tanh_kernel(float* __restrict__ data, const float* __restrict__ bias, 
                                       int N, int C, int H, int W) {
    int total_elements = N * C * H * W;
    int hw = H * W;
    
    // Vectorized processing: each thread handles 4 elements
    int total_vectors = (total_elements + 3) / 4;
    
    // Grid-stride loop over vectors
    for (int vec_idx = blockIdx.x * blockDim.x + threadIdx.x; vec_idx < total_vectors; vec_idx += blockDim.x * gridDim.x) {
        // Calculate base index for this vector
        int base_idx = vec_idx * 4;
        
        // Load 4 floats in a coalesced manner
        float4* data_vec_ptr = reinterpret_cast<float4*>(data);
        float4 data_vec = data_vec_ptr[vec_idx];
        
        // Process each element in the vector
        float vals[4] = {data_vec.x, data_vec.y, data_vec.z, data_vec.w};
        
        // Apply bias subtraction and tanh for each element
        for (int j = 0; j < 4; ++j) {
            if (base_idx + j < total_elements) {
                int i = base_idx + j;
                int c = (i / hw) % C;
                vals[j] = vals[j] - bias[c];
                vals[j] = tanhf(vals[j]);
            }
        }
        
        // Store back the results
        data_vec.x = vals[0];
        data_vec.y = vals[1];
        data_vec.z = vals[2];
        data_vec.w = vals[3];
        data_vec_ptr[vec_idx] = data_vec;
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor bias) {
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    
    int total_elements = N * C * H * W;
    int threads = 256;
    // Calculate number of vectors instead of elements
    int total_vectors = (total_elements + 3) / 4;
    int blocks = (total_vectors + threads - 1) / threads;
    // Cap blocks to avoid excessive resource usage
    blocks = std::min(blocks, 65535);

    fused_bias_tanh_kernel<<<blocks, threads>>>(x.data_ptr<float>(), bias.data_ptr<float>(), N, C, H, W);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor x, torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Bias Subtraction and Tanh with Vectorization");
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

# Custom CUDA kernel for transposed convolution
conv_transpose_cuda = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
) {
    // Calculate global thread index
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * out_height * out_width;
    
    if (out_idx >= total_outputs) return;
    
    // Decompose output index
    int b = out_idx / (out_channels * out_height * out_width);
    int c_out = (out_idx / (out_height * out_width)) % out_channels;
    int h_out = (out_idx / out_width) % out_height;
    int w_out = out_idx % out_width;
    
    float sum = 0.0f;
    
    // Group handling
    int group_id = c_out * groups / out_channels;
    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;
    
    // Iterate through input channels in this group
    for (int c_in_group = 0; c_in_group < in_channels_per_group; ++c_in_group) {
        int c_in = group_id * in_channels_per_group + c_in_group;
        
        // Iterate through kernel
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                // Calculate corresponding input position
                int h_in = h_out + padding - kh * dilation;
                int w_in = w_out + padding - kw * dilation;
                
                // Check if within input bounds and stride alignment
                if (h_in % stride == 0 && w_in % stride == 0) {
                    h_in /= stride;
                    w_in /= stride;
                    
                    if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                        // Calculate indices
                        int input_idx = b * (in_channels * in_height * in_width) + 
                                        c_in * (in_height * in_width) + 
                                        h_in * in_width + w_in;
                                        
                        int weight_idx = c_in_group * (out_channels_per_group * kernel_size * kernel_size) +
                                         (c_out % out_channels_per_group) * (kernel_size * kernel_size) +
                                         kh * kernel_size + kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add bias
    sum += bias[c_out];
    
    // Store result
    output[out_idx] = sum;
}

void conv_transpose_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_channels = weight.size(1); // Note: transposed conv weight shape is (in_channels, out_channels/groups, kH, kW)
    int kernel_size = weight.size(2);
    
    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    
    int total_outputs = batch_size * out_channels * out_height * out_width;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    blocks = std::min(blocks, 65535);
    
    conv_transpose_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        dilation
    );
}
"""

conv_transpose_cpp = r"""
#include <torch/extension.h>

void conv_transpose_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose", &conv_transpose_forward, "Custom Conv Transpose Forward");
}
"""

# Compile the conv transpose extension
conv_transpose_ext = load_inline(
    name='conv_transpose_op',
    cpp_sources=conv_transpose_cpp,
    cuda_sources=conv_transpose_cuda,
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
    # 1. Custom Conv Transpose implementation
    # Calculate output dimensions
    in_channels = x.size(1)
    in_height = x.size(2)
    in_width = x.size(3)
    out_channels = conv_transpose_weight.size(1)
    kernel_size = conv_transpose_weight.size(2)
    
    out_height = (in_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + conv_transpose_output_padding + 1
    out_width = (in_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + conv_transpose_output_padding + 1
    
    # Create output tensor
    output = torch.empty(
        (x.size(0), out_channels, out_height, out_width),
        dtype=x.dtype,
        device=x.device
    )
    
    # Run custom conv transpose kernel
    conv_transpose_ext.conv_transpose(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        output,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_groups,
        conv_transpose_dilation
    )
    
    # 2. Fused operation using our vectorized CUDA kernel
    # Convert bias (C, 1, 1) to contiguous 1D for the kernel
    bias_flat = bias.view(-1).contiguous()
    output = output.contiguous()
    fused_ext.fused_op(output, bias_flat)
    
    return output

# Placeholder parameters as defined in the prompt
batch_size = 32
in_channels  = 64  
out_channels = 64  
height = width = 256 
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
