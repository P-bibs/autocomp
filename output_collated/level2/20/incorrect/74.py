# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_140117/code_9.py
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

# -------------------------------------------------------------------------
# Optimized CUDA kernel – fused transposed convolution + post-op
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE_DIM 32
#define CHANNEL_TILE 32

__global__ void fused_transposed_conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ post_bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_d, const int input_h, const int input_w,
    const int output_d, const int output_h, const int output_w,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int padding_d, const int padding_h, const int padding_w,
    const int output_padding_d, const int output_padding_h, const int output_padding_w,
    const int dilation_d, const int dilation_h, const int dilation_w
) {
    // Shared memory for weight tile
    extern __shared__ float s_weight[];

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_z = blockIdx.z;

    if (out_x >= output_w || out_y >= output_h || out_z >= output_d) return;

    int spatial_out_size = output_d * output_h * output_w;
    int channel_out_size = out_channels * spatial_out_size;

    for (int batch = 0; batch < batch_size; batch++) {
        for (int out_c = 0; out_c < out_channels; out_c++) {
            float sum = (conv_bias) ? __ldg(&conv_bias[out_c]) : 0.0f;

            // Loop over input and kernel
            for (int in_c = 0; in_c < in_channels; in_c++) {
                for (int kd = 0; kd < kernel_d; kd++) {
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            // Calculate corresponding input position
                            int in_z = out_z - stride_d * kd + padding_d - output_padding_d;
                            int in_y = out_y - stride_h * kh + padding_h - output_padding_h;
                            int in_x = out_x - stride_w * kw + padding_w - output_padding_w;

                            // Check if divisible by dilation
                            if (in_z % dilation_d == 0 && in_y % dilation_h == 0 && in_x % dilation_w == 0) {
                                in_z /= dilation_d;
                                in_y /= dilation_h;
                                in_x /= dilation_w;

                                // Check bounds
                                if (in_z >= 0 && in_z < input_d && in_y >= 0 && in_y < input_h && in_x >= 0 && in_x < input_w) {
                                    int input_idx = batch * (in_channels * input_d * input_h * input_w) +
                                                    in_c * (input_d * input_h * input_w) +
                                                    in_z * (input_h * input_w) +
                                                    in_y * input_w +
                                                    in_x;

                                    int weight_idx = out_c * (in_channels * kernel_d * kernel_h * kernel_w) +
                                                     in_c * (kernel_d * kernel_h * kernel_w) +
                                                     kd * (kernel_h * kernel_w) +
                                                     kh * kernel_w +
                                                     kw;

                                    sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                                }
                            }
                        }
                    }
                }
            }

            // Apply post-op: ((x + bias) + x) * x + x = 2*x*x + bias*x + x
            float bias_val = (post_bias) ? __ldg(&post_bias[out_c]) : 0.0f;
            float tmp = sum + bias_val;
            float tmp2 = tmp + sum;
            float result = tmp2 * sum + sum;

            int output_idx = batch * channel_out_size +
                             out_c * spatial_out_size +
                             out_z * (output_h * output_w) +
                             out_y * output_w +
                             out_x;

            output[output_idx] = result;
        }
    }
}

void fused_transposed_conv3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& output_padding,
    const std::vector<int64_t>& dilation
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_d = input.size(2);
    const int input_h = input.size(3);
    const int input_w = input.size(4);

    const int out_channels = weight.size(1);
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);

    const int output_d = output.size(2);
    const int output_h = output.size(3);
    const int output_w = output.size(4);

    dim3 block(8, 8, 1);
    dim3 grid(
        (output_w + block.x - 1) / block.x,
        (output_h + block.y - 1) / block.y,
        output_d
    );

    size_t shared_mem_size = kernel_d * kernel_h * kernel_w * sizeof(float);

    fused_transposed_conv3d_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        post_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_d, kernel_h, kernel_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        dilation[0], dilation[1], dilation[2]
    );
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
#include <vector>

void fused_transposed_conv3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& output_padding,
    const std::vector<int64_t>& dilation
);

torch::Tensor fused_transposed_conv3d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& output_padding,
    const std::vector<int64_t>& dilation
) {
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    if (conv_bias.defined()) TORCH_CHECK(conv_bias.is_contiguous(), "conv_bias must be contiguous");
    if (post_bias.defined()) TORCH_CHECK(post_bias.is_contiguous(), "post_bias must be contiguous");

    // Calculate output dimensions
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    std::vector<int64_t> output_sizes(5);
    output_sizes[0] = input_sizes[0];  // N
    output_sizes[1] = weight_sizes[1]; // C_out
    
    output_sizes[2] = (input_sizes[2] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (weight_sizes[2] - 1) + output_padding[0] + 1; // D
    output_sizes[3] = (input_sizes[3] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (weight_sizes[3] - 1) + output_padding[1] + 1; // H
    output_sizes[4] = (input_sizes[4] - 1) * stride[2] - 2 * padding[2] + dilation[2] * (weight_sizes[4] - 1) + output_padding[2] + 1; // W

    auto output = torch::empty(output_sizes, input.options());
    
    fused_transposed_conv3d_forward(
        input, weight, conv_bias, post_bias, output,
        stride, padding, output_padding, dilation
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_transposed_conv3d", &fused_transposed_conv3d, "Fused 3D Transposed Convolution with Post-op");
}
"""

# -------------------------------------------------------------------------
# Build the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_transposed_conv3d_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Optimized functional model
# -------------------------------------------------------------------------
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
    # Ensure groups is 1 as we don't support grouped conv in our kernel
    if conv_transpose_groups != 1:
        raise NotImplementedError("Grouped transposed convolution not supported in custom kernel")
    
    # Convert parameters to lists if they're tuples
    if isinstance(conv_transpose_stride, int):
        conv_transpose_stride = [conv_transpose_stride] * 3
    if isinstance(conv_transpose_padding, int):
        conv_transpose_padding = [conv_transpose_padding] * 3
    if isinstance(conv_transpose_output_padding, int):
        conv_transpose_output_padding = [conv_transpose_output_padding] * 3
    if isinstance(conv_transpose_dilation, int):
        conv_transpose_dilation = [conv_transpose_dilation] * 3
    
    # Flatten bias to a 1-D tensor (required by the kernel)
    bias_flat = bias.view(-1) if bias is not None else torch.tensor([], device=x.device, dtype=x.dtype)
    
    # Call our fused kernel
    return fused_ext.fused_transposed_conv3d(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        bias_flat,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_dilation
    )

# -------------------------------------------------------------------------
# Helper code (shape parameters, input factories)
# -------------------------------------------------------------------------
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
