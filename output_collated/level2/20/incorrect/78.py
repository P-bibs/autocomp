# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_140117/code_14.py
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

# ----------------------------------------------------------------------
# Optimised CUDA kernel: bias is cached in shared memory
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void fused_post_conv_kernel(
    const float4* __restrict__ input,
    const float* __restrict__ bias,      // original bias in global memory
    float4* __restrict__ output,
    int64_t num_elements_float4,
    int64_t spatial_size,
    int64_t out_channels
) {
    // ---- shared‑memory cache for bias ---------------------------------
    extern __shared__ float bias_cache[];

    // Load bias from global memory into shared memory (once per block)
    if (threadIdx.x < out_channels) {
        bias_cache[threadIdx.x] = bias[threadIdx.x];
    }
    __syncthreads();

    // ---- grid‑stride loop --------------------------------------------
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < num_elements_float4;
         idx += blockDim.x * gridDim.x) {

        // channel index for the first element of the float4 vector
        int64_t base_channel_idx = (idx * 4 / spatial_size) % out_channels;

        // load 4 consecutive elements
        float4 x_vec = input[idx];
        float4 res;

        // read bias from shared memory (no global memory traffic)
        float b = bias_cache[base_channel_idx];

        // original arithmetic: ((x + b) + x) * x + x
        res.x = ((x_vec.x + b) + x_vec.x) * x_vec.x + x_vec.x;
        res.y = ((x_vec.y + b) + x_vec.y) * x_vec.y + x_vec.y;
        res.z = ((x_vec.z + b) + x_vec.z) * x_vec.z + x_vec.z;
        res.w = ((x_vec.w + b) + x_vec.w) * x_vec.w + x_vec.w;

        output[idx] = res;
    }
}

// CUDA kernel for 3D transposed convolution
__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
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
    int output_padding
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * out_depth * out_height * out_width;

    if (out_idx >= total_output_elements) return;

    int tmp = out_idx;
    int w = tmp % out_width;
    tmp /= out_width;
    int h = tmp % out_height;
    tmp /= out_height;
    int d = tmp % out_depth;
    tmp /= out_depth;
    int c = tmp % out_channels;
    int n = tmp / out_channels;

    float val = 0.0f;

    // Loop over kernel dimensions
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int in_d = d - kd * stride + padding;
                int in_h = h - kh * stride + padding;
                int in_w = w - kw * stride + padding;

                if (in_d % stride == 0 && in_h % stride == 0 && in_w % stride == 0) {
                    in_d /= stride;
                    in_h /= stride;
                    in_w /= stride;

                    if (in_d >= 0 && in_d < in_depth &&
                        in_h >= 0 && in_h < in_height &&
                        in_w >= 0 && in_w < in_width) {
                        for (int ic = 0; ic < in_channels; ic++) {
                            int input_idx = n * (in_channels * in_depth * in_height * in_width) +
                                            ic * (in_depth * in_height * in_width) +
                                            in_d * (in_height * in_width) +
                                            in_h * in_width + in_w;
                            int weight_idx = c * (in_channels * kernel_size * kernel_size * kernel_size) +
                                             ic * (kernel_size * kernel_size * kernel_size) +
                                             kd * (kernel_size * kernel_size) +
                                             kh * kernel_size + kw;
                            val += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        val += bias[c];
    }

    output[out_idx] = val;
}

void fused_post_conv_forward(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output
) {
    const int64_t num_elements      = input.numel();
    const int64_t num_elements_float4 = (num_elements + 3) / 4;          // ceiling
    const int64_t spatial_size      = input.size(2) * input.size(3) * input.size(4);
    const int64_t out_channels      = input.size(1);

    // choose thread count (multiple of 32)
    const int threads_per_block = 256;

    // simple block count that saturates the device for large tensors
    int blocks = static_cast<int>((num_elements_float4 + threads_per_block - 1) / threads_per_block);
    if (blocks > 1024) blocks = 1024;   // cap to keep launch overhead small

    // shared memory size = out_channels * sizeof(float)
    const size_t shared_mem = out_channels * sizeof(float);

    // cast to float4 pointers for vectorised loads / stores
    const float4* in_ptr  = reinterpret_cast<const float4*>(input.data_ptr<float>());
    float4*       out_ptr = reinterpret_cast<float4*>(output.data_ptr<float>());

    fused_post_conv_kernel<<<blocks, threads_per_block, shared_mem>>>(
        in_ptr,
        bias.data_ptr<float>(),
        out_ptr,
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
    int padding,
    int output_padding
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    int out_channels = output.size(1);
    int out_depth = output.size(2);
    int out_height = output.size(3);
    int out_width = output.size(4);
    
    int kernel_size = weight.size(2); // Assuming cubic kernel
    
    int total_output_elements = batch_size * out_channels * out_depth * out_height * out_width;
    int threads_per_block = 256;
    int blocks = (total_output_elements + threads_per_block - 1) / threads_per_block;
    
    conv_transpose3d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
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
        output_padding
    );
}
"""

# ----------------------------------------------------------------------
# C++ binding (PYBIND11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_post_conv_forward(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output);

void conv_transpose3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding);

torch::Tensor fused_post_conv(const torch::Tensor& input, const torch::Tensor& bias) {
    auto output = torch::empty_like(input);
    fused_post_conv_forward(input, bias, output);
    return output;
}

torch::Tensor conv_transpose3d_custom(const torch::Tensor& input, 
                                      const torch::Tensor& weight,
                                      const torch::Tensor& bias,
                                      int stride,
                                      int padding,
                                      int output_padding) {
    // Calculate output dimensions
    int out_depth = (input.size(2) - 1) * stride - 2 * padding + weight.size(2) + output_padding;
    int out_height = (input.size(3) - 1) * stride - 2 * padding + weight.size(3) + output_padding;
    int out_width = (input.size(4) - 1) * stride - 2 * padding + weight.size(4) + output_padding;
    
    auto output = torch::empty({input.size(0), weight.size(0), out_depth, out_height, out_width}, 
                               torch::dtype(input.dtype()).device(input.device()));
    
    conv_transpose3d_forward(input, weight, bias, output, stride, padding, output_padding);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_conv", &fused_post_conv,
          "Fused post‑conv arithmetic with float4 vectorisation and shared‑memory bias");
    m.def("conv_transpose3d_custom", &conv_transpose3d_custom,
          "Custom 3D transposed convolution");
}
"""

# ----------------------------------------------------------------------
# Compile the inline extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_post_conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional model used by the benchmark
# ----------------------------------------------------------------------
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
    # 3‑D transposed convolution with custom CUDA kernel
    x = fused_ext.conv_transpose3d_custom(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        conv_transpose_stride[0],  # Assuming uniform stride
        conv_transpose_padding[0],  # Assuming uniform padding
        conv_transpose_output_padding[0]  # Assuming uniform output padding
    )

    # flatten the secondary bias so that it can be passed to the custom kernel
    bias_flat = bias.view(-1)

    # call the optimized fused kernel (bias is cached in shared memory)
    return fused_ext.fused_post_conv(x, bias_flat)


# ----------------------------------------------------------------------
# Helper functions to create inputs (identical to the original benchmark)
# ----------------------------------------------------------------------
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
