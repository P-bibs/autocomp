# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_133559/code_14.py
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

# Optimized CUDA kernel using float4 vectorization, shared memory for bias, and incremental channel indexing
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define FMA(a, b, c) fma(a, b, c)

__global__ void fused_post_conv_kernel(
    const float4* __restrict__ input,
    const float* __restrict__ bias,
    float4* __restrict__ output,
    int64_t num_elements_float4,
    int64_t spatial_size,
    int64_t out_channels
) {
    // Shared memory for bias caching
    extern __shared__ float shared_bias[];
    
    // Cooperative loading of bias into shared memory
    int tid = threadIdx.x;
    for (int i = tid; i < out_channels; i += blockDim.x) {
        shared_bias[i] = bias[i];
    }
    __syncthreads();
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements_float4) {
        // Base linear index of the first element in this float4 group
        int64_t base_idx = idx * 4;

        // Initial batch*channel index and channel (computed once per thread)
        int64_t bc        = base_idx / spatial_size;
        int64_t channel   = bc % out_channels;
        int64_t spatial   = base_idx % spatial_size;

        // Load four input elements as a vector
        float4 x_vec = input[idx];
        float4 res;

        // ----- element 0 -----
        float x0 = x_vec.x;
        float b0 = shared_bias[channel];
        float t0 = x0 + x0;                     // 2*x0
        res.x = FMA(x0, t0 + b0, x0);           // (2*x0 + b0) * x0 + x0

        // Advance spatial index; wrap to next channel when needed
        spatial++;
        if (spatial == spatial_size) {
            spatial = 0;
            channel++;
            if (channel == out_channels) channel = 0;
        }

        // ----- element 1 -----
        float x1 = x_vec.y;
        float b1 = shared_bias[channel];
        float t1 = x1 + x1;
        res.y = FMA(x1, t1 + b1, x1);

        spatial++;
        if (spatial == spatial_size) {
            spatial = 0;
            channel++;
            if (channel == out_channels) channel = 0;
        }

        // ----- element 2 -----
        float x2 = x_vec.z;
        float b2 = shared_bias[channel];
        float t2 = x2 + x2;
        res.z = FMA(x2, t2 + b2, x2);

        spatial++;
        if (spatial == spatial_size) {
            spatial = 0;
            channel++;
            if (channel == out_channels) channel = 0;
        }

        // ----- element 3 -----
        float x3 = x_vec.w;
        float b3 = shared_bias[channel];
        float t3 = x3 + x3;
        res.w = FMA(x3, t3 + b3, x3);

        // Store the result
        output[idx] = res;
    }
}

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
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
    int padding
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (out_idx >= total_output_elements) return;
    
    int w = out_idx % output_width;
    out_idx /= output_width;
    int h = out_idx % output_height;
    out_idx /= output_height;
    int d = out_idx % output_depth;
    out_idx /= output_depth;
    int c = out_idx % out_channels;
    int b = out_idx / out_channels;
    
    float sum = 0.0f;
    
    // Calculate input position corresponding to output position
    int start_kd = (d + padding - kernel_size + 1 + stride - 1) / stride;  // ceil division
    int start_kh = (h + padding - kernel_size + 1 + stride - 1) / stride;
    int start_kw = (w + padding - kernel_size + 1 + stride - 1) / stride;
    
    int end_kd = min((d + padding) / stride + 1, kernel_size);  // floor division
    int end_kh = min((h + padding) / stride + 1, kernel_size);
    int end_kw = min((w + padding) / stride + 1, kernel_size);
    
    start_kd = max(start_kd, 0);
    start_kh = max(start_kh, 0);
    start_kw = max(start_kw, 0);
    
    for (int kd = start_kd; kd < end_kd; kd++) {
        for (int kh = start_kh; kh < end_kh; kh++) {
            for (int kw = start_kw; kw < end_kw; kw++) {
                int id = d + padding - kd * stride;
                int ih = h + padding - kh * stride;
                int iw = w + padding - kw * stride;
                
                if (id >= 0 && id < input_depth && ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    for (int ic = 0; ic < in_channels; ic++) {
                        int input_idx = ((((b * in_channels + ic) * input_depth + id) * input_height + ih) * input_width + iw);
                        int weight_idx = ((((c * in_channels + ic) * kernel_size + kd) * kernel_size + kh) * kernel_size + kw);
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    output[out_idx * output_depth * output_height * output_width + d * output_height * output_width + h * output_width + w] = sum + bias[c];
}

void fused_post_conv_forward(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output
) {
    // Calculate number of float4 elements (total elements / 4)
    int64_t num_elements = input.numel();
    int64_t num_elements_float4 = num_elements / 4;  // Exact division since we ensure alignment
    
    int64_t spatial_size = input.size(2) * input.size(3) * input.size(4);
    int64_t out_channels = input.size(1);
    
    int threads_per_block = 256;
    int blocks = (num_elements_float4 + threads_per_block - 1) / threads_per_block;
    
    // Cast pointers to float4 for vectorized access
    const float4* input_ptr = reinterpret_cast<const float4*>(input.data_ptr<float>());
    float4* output_ptr = reinterpret_cast<float4*>(output.data_ptr<float>());
    
    // Shared memory size for bias
    size_t shared_mem_size = out_channels * sizeof(float);
    
    fused_post_conv_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        input_ptr,
        bias.data_ptr<float>(),
        output_ptr,
        num_elements_float4,
        spatial_size,
        out_channels
    );
}

void conv_transpose3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);
    
    int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size;
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size;
    
    int total_output_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    int threads_per_block = 256;
    int blocks = (total_output_elements + threads_per_block - 1) / threads_per_block;
    
    conv_transpose3d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
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
        padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_post_conv_forward(const torch::Tensor& input, const torch::Tensor& bias, torch::Tensor& output);
void conv_transpose3d_forward(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, 
                              torch::Tensor& output, int stride, int padding);

torch::Tensor fused_post_conv(const torch::Tensor& input, const torch::Tensor& bias) {
    auto output = torch::empty_like(input);
    fused_post_conv_forward(input, bias, output);
    return output;
}

torch::Tensor custom_conv_transpose3d(const torch::Tensor& input, const torch::Tensor& weight, 
                                      const torch::Tensor& bias, int stride, int padding) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);
    
    int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size;
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size;
    
    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    conv_transpose3d_forward(input, weight, bias, output, stride, padding);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_conv", &fused_post_conv, "Fused post-conv arithmetic with float4 vectorization and shared memory");
    m.def("custom_conv_transpose3d", &custom_conv_transpose3d, "Custom conv transpose 3d implementation");
}
"""

fused_ext = load_inline(
    name='fused_post_conv_ext',
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
    # Perform the convolution using our custom CUDA implementation
    x = fused_ext.custom_conv_transpose3d(x, conv_transpose_weight, conv_transpose_bias, 
                                         conv_transpose_stride, conv_transpose_padding)
    
    # Flatten bias for simplified kernel indexing
    bias_flat = bias.view(-1)
    
    # Ensure input is contiguous and properly aligned for float4 access
    x = x.contiguous()
    
    # Use optimized fused kernel for the intensive post-processing element-wise ops
    return fused_ext.fused_post_conv(x, bias_flat)

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
