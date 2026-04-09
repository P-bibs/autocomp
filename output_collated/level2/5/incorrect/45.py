# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_121229/code_5.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <math.h>

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int dilation
) {
    int out_ch = blockIdx.x;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    
    // Shared memory for weights
    extern __shared__ float shared_weights[];
    
    // Load weights into shared memory
    int weights_per_thread = (in_channels * kernel_size * kernel_size + total_threads - 1) / total_threads;
    for (int i = 0; i < weights_per_thread; ++i) {
        int idx = tid + i * total_threads;
        if (idx < in_channels * kernel_size * kernel_size) {
            shared_weights[idx] = weight[out_ch * in_channels * kernel_size * kernel_size + idx];
        }
    }
    __syncthreads();
    
    // Each block handles one output channel
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int out_y = threadIdx.x; out_y < output_height; out_y += blockDim.x) {
            for (int out_x = 0; out_x < output_width; ++out_x) {
                float sum = 0.0f;
                
                for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
                    for (int k_y = 0; k_y < kernel_size; ++k_y) {
                        for (int k_x = 0; k_x < kernel_size; ++k_x) {
                            int in_y = out_y - k_y * dilation + padding;
                            int in_x = out_x - k_x * dilation + padding;
                            
                            if ((in_y % stride == 0) && (in_x % stride == 0)) {
                                in_y /= stride;
                                in_x /= stride;
                                
                                if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                                    int input_idx = batch * in_channels * input_height * input_width +
                                                    in_ch * input_height * input_width +
                                                    in_y * input_width + in_x;
                                    int weight_idx = in_ch * kernel_size * kernel_size + 
                                                     (kernel_size - 1 - k_y) * kernel_size + (kernel_size - 1 - k_x);
                                    sum += input[input_idx] * shared_weights[weight_idx];
                                }
                            }
                        }
                    }
                }
                
                int output_idx = batch * out_channels * output_height * output_width +
                                 out_ch * output_height * output_width +
                                 out_y * output_width + out_x;
                output[output_idx] = sum + bias[out_ch];
            }
        }
    }
}

__global__ void fused_op_vectorized_kernel(
    float* __restrict__ data, 
    const float* __restrict__ bias, 
    int total_elements, 
    int C, 
    int HW
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    // Process float4 chunks
    if (idx + 3 < total_elements) {
        float4* data_ptr = reinterpret_cast<float4*>(&data[idx]);
        float4 val = *data_ptr;
        
        // Unroll processing of 4 floats
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int current_idx = idx + i;
            int c = (current_idx / HW) % C;
            float b = __ldg(&bias[c]); // Use read-only cache for bias
            
            if (i == 0) val.x = tanhf(val.x - b);
            else if (i == 1) val.y = tanhf(val.y - b);
            else if (i == 2) val.z = tanhf(val.z - b);
            else val.w = tanhf(val.w - b);
        }
        *data_ptr = val;
    } else {
        // Cleanup for remaining elements
        for (int i = idx; i < total_elements; ++i) {
            int c = (i / HW) % C;
            data[i] = tanhf(data[i] - bias[c]);
        }
    }
}

void fused_op_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    torch::Tensor& output,
    const torch::Tensor& fused_bias,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int dilation
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    
    int out_channels = weight.size(0);
    int output_height = (input_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    
    // Conv transpose
    int threads = 256;
    int blocks = out_channels;
    size_t shared_mem_size = in_channels * kernel_size * kernel_size * sizeof(float);
    
    conv_transpose2d_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation
    );
    
    // Fused op
    int HW = output_height * output_width;
    int total_elements = batch_size * out_channels * HW;
    
    int fused_threads = 256;
    int fused_blocks = (total_elements / 4 + fused_threads - 1) / fused_threads;
    
    fused_op_vectorized_kernel<<<fused_blocks, fused_threads>>>(
        output.data_ptr<float>(), 
        fused_bias.data_ptr<float>(), 
        total_elements, 
        out_channels, 
        HW
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    torch::Tensor& output,
    const torch::Tensor& fused_bias,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op_forward", &fused_op_forward, "Fused conv transpose and bias-tanh operation");
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
    # Assume groups=1 as per typical usage
    batch_size, in_channels, height, width = x.shape
    out_channels = conv_transpose_weight.shape[0]
    kernel_size = conv_transpose_weight.shape[2]
    
    # Calculate output dimensions
    output_height = (height - 1) * conv_transpose_stride + conv_transpose_output_padding - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + 1
    output_width = (width - 1) * conv_transpose_stride + conv_transpose_output_padding - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=x.device, dtype=x.dtype)
    
    # Run fused kernel
    fused_ext.fused_op_forward(
        x.contiguous(),
        conv_transpose_weight.contiguous(),
        conv_transpose_bias.contiguous(),
        output,
        bias.view(-1).contiguous(),
        kernel_size,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_dilation
    )
    
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
