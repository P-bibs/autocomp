# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_121229/code_9.py
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

# CUDA Kernel: Vectorized with float4 operations and custom conv transpose
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_op_kernel_vectorized(
    float* __restrict__ data, 
    const float* __restrict__ bias, 
    int N, int C, int HW,
    int total_elements_scalar
) {
    // Shared memory for bias caching
    extern __shared__ float shared_bias[];
    
    // Cooperatively load bias into shared memory
    int bias_load_idx = threadIdx.x;
    while (bias_load_idx < C) {
        shared_bias[bias_load_idx] = bias[bias_load_idx];
        bias_load_idx += blockDim.x;
    }
    __syncthreads();
    
    // Precompute constants
    int C_HW = C * HW;
    
    // Cast to float4 for vectorized operations
    float4* data_vec = reinterpret_cast<float4*>(data);
    int total_elements_vec = (total_elements_scalar + 3) / 4;
    
    // Grid-stride loop over float4 elements
    int idx_vec = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (idx_vec < total_elements_vec) {
        int idx_scalar = idx_vec * 4;
        
        // Process up to 4 elements
        float4 vals = data_vec[idx_vec];
        
        #pragma unroll 4
        for (int i = 0; i < 4 && (idx_scalar + i) < total_elements_scalar; ++i) {
            int scalar_idx = idx_scalar + i;
            
            // Efficient index computation
            int n = scalar_idx / C_HW;
            int remainder = scalar_idx % C_HW;
            int c = remainder / HW;
            
            float* val_ptr;
            if (i == 0) val_ptr = &vals.x;
            else if (i == 1) val_ptr = &vals.y;
            else if (i == 2) val_ptr = &vals.z;
            else val_ptr = &vals.w;
            
            *val_ptr = tanhf(*val_ptr - shared_bias[c]);
        }
        
        // Write back coalesced
        data_vec[idx_vec] = vals;
        
        idx_vec += gridDim.x * blockDim.x;
    }
}

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out_elements = batch_size * out_channels * out_height * out_width;
    
    if (out_idx >= total_out_elements) return;
    
    int n = out_idx / (out_channels * out_height * out_width);
    int remainder = out_idx % (out_channels * out_height * out_width);
    int c = remainder / (out_height * out_width);
    remainder = remainder % (out_height * out_width);
    int h = remainder / out_width;
    int w = remainder % out_width;
    
    float sum = 0.0f;
    
    // Compute convolution
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int in_h = h - kh * stride + padding;
                int in_w = w - kw * stride + padding;
                
                // Check bounds with output padding consideration
                if (in_h >= 0 && in_h < in_height * stride && 
                    in_w >= 0 && in_w < in_width * stride &&
                    in_h % stride == 0 && in_w % stride == 0) {
                    
                    in_h /= stride;
                    in_w /= stride;
                    
                    if (in_h < in_height && in_w < in_width) {
                        int input_idx = ((n * in_channels + ic) * in_height + in_h) * in_width + in_w;
                        int weight_idx = ((c * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add bias
    sum += bias[c];
    output[out_idx] = sum;
}

void fused_op_forward(torch::Tensor& x, const torch::Tensor& bias) {
    int N = x.size(0);
    int C = x.size(1);
    int HW = x.size(2) * x.size(3);
    int total_elements = N * C * HW;
    
    int threads = 256;
    
    // Process float4 elements
    int blocks = ((total_elements + 3) / 4 + threads - 1) / threads;
    blocks = std::min(blocks, 65536);
    
    size_t shared_mem_size = C * sizeof(float);
    
    fused_op_kernel_vectorized<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        N, C, HW,
        total_elements
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void custom_conv_transpose2d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);
    
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    int total_out_elements = batch_size * out_channels * out_height * out_width;
    int threads = 256;
    int blocks = (total_out_elements + threads - 1) / threads;
    blocks = std::min(blocks, 65536);
    
    conv_transpose2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        output_padding
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor& x, const torch::Tensor& bias);
void custom_conv_transpose2d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op_forward", &fused_op_forward, "Fused bias subtraction and tanh with float4 vectorization");
    m.def("conv_transpose2d", &custom_conv_transpose2d, "Custom conv transpose 2d implementation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_ext',
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
):
    # Create output tensor with correct size
    batch_size = x.size(0)
    out_channels = conv_transpose_weight.size(1)
    kernel_size = conv_transpose_weight.size(2)
    
    # Calculate output dimensions
    out_height = (x.size(2) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    out_width = (x.size(3) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Perform convolution transpose using custom CUDA kernel
    fused_ext.conv_transpose2d(
        x, conv_transpose_weight, conv_transpose_bias, output,
        conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding
    )
    
    # Flatten bias for kernel usage
    bias_flat = bias.view(-1)
    
    # Run optimized kernel with float4 vectorization
    fused_ext.fused_op_forward(output, bias_flat)
    
    return output

# Setup for testing
batch_size = 32
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]
