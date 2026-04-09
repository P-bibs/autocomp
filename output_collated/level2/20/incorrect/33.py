# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_132207/code_0.py
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
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------------------------
# Optimized CUDA kernel – fuses transposed convolution + activation
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#define THREADS_PER_BLOCK 256
#define ELEMENTS_PER_THREAD 4

inline __device__ float fused_activation(float x, float bias_val) {
    // ((x + bias) + x) * x + x = 2*x*x + bias*x + x
    float tmp = x + bias_val;        // x + b
    float tmp2 = tmp + x;            // (x + b) + x = 2*x + b
    return tmp2 * x + x;             // ((2*x + b) * x) + x
}

__global__ void fused_transposed_conv3d_activation_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ post_bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int id, const int ih, const int iw,
    const int od, const int oh, const int ow,
    const int kd, const int kh, const int kw,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int out_pad_d, const int out_pad_h, const int out_pad_w,
    const int dilation_d, const int dilation_h, const int dilation_w
) {
    // Global thread index
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;
    const int total_elements = batch_size * out_channels * od * oh * ow;

    if (idx >= total_elements) return;

    // Precompute strides for output tensor
    const int ow_stride = 1;
    const int oh_stride = ow;
    const int od_stride = oh * ow;
    const int oc_stride = od * oh * ow;
    const int ob_stride = out_channels * oc_stride;

    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        if (idx + i >= total_elements) break;

        // Decode output index (n, c, odx, ohy, owz)
        int temp = idx + i;
        const int oz = (temp / ow_stride) % ow;
        const int oy = (temp / oh_stride) % oh;
        const int ox = (temp / od_stride) % od;
        const int oc = (temp / oc_stride) % out_channels;
        const int ob = temp / ob_stride;

        // Compute input spatial coordinates
        const int id_start = ox * stride_d - pad_d;
        const int ih_start = oy * stride_h - pad_h;
        const int iw_start = oz * stride_w - pad_w;

        float sum = 0.0f;

        // Iterate over kernel size
        for (int kd_idx = 0; kd_idx < kd; ++kd_idx) {
            for (int kh_idx = 0; kh_idx < kh; ++kh_idx) {
                for (int kw_idx = 0; kw_idx < kw; ++kw_idx) {
                    const int in_d = id_start + kd_idx * dilation_d;
                    const int in_h = ih_start + kh_idx * dilation_h;
                    const int in_w = iw_start + kw_idx * dilation_w;

                    if (in_d >= 0 && in_d < id &&
                        in_h >= 0 && in_h < ih &&
                        in_w >= 0 && in_w < iw) {

                        // Weight index calculation
                        const int weight_base = oc * (in_channels * kd * kh * kw) +
                                                kd_idx * (kh * kw) +
                                                kh_idx * kw +
                                                kw_idx;

                        // Input index calculation
                        const int input_base = ob * (in_channels * id * ih * iw) +
                                               in_d * (ih * iw) +
                                               in_h * iw +
                                               in_w;

                        // Channel loop vectorized manually
                        for (int ic = 0; ic < in_channels; ++ic) {
                            const int input_idx = input_base + ic * (id * ih * iw);
                            const int weight_idx = weight_base + ic * (kd * kh * kw);
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }

        // Add bias
        sum += conv_bias[oc];

        // Apply fused activation
        float result = fused_activation(sum, post_bias[oc]);

        output[idx + i] = result;
    }
}

void fused_transposed_conv3d_activation_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    const auto input_sizes = input.sizes();
    const auto weight_sizes = weight.sizes();
    const auto output_sizes = output.sizes();

    const int batch_size = input_sizes[0];
    const int in_channels = input_sizes[1];
    const int id = input_sizes[2], ih = input_sizes[3], iw = input_sizes[4];
    const int out_channels = weight_sizes[0];
    const int kd = weight_sizes[2], kh = weight_sizes[3], kw = weight_sizes[4];
    const int od = output_sizes[2], oh = output_sizes[3], ow = output_sizes[4];

    const int total_elements = batch_size * out_channels * od * oh * ow;
    const int blocks = (total_elements + THREADS_PER_BLOCK * ELEMENTS_PER_THREAD - 1) / (THREADS_PER_BLOCK * ELEMENTS_PER_THREAD);

    at::cuda::CUDAGuard device_guard(input.device());
    fused_transposed_conv3d_activation_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        post_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        id, ih, iw, od, oh, ow,
        kd, kh, kw,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_pad_d, out_pad_h, out_pad_w,
        dilation_d, dilation_h, dilation_w
    );
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_transposed_conv3d_activation_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w,
    int dilation_d, int dilation_h, int dilation_w);

torch::Tensor fused_transposed_conv3d_activation(const torch::Tensor& input,
                                                 const torch::Tensor& weight,
                                                 const torch::Tensor& conv_bias,
                                                 const torch::Tensor& post_bias,
                                                 std::vector<int64_t> stride,
                                                 std::vector<int64_t> padding,
                                                 std::vector<int64_t> output_padding,
                                                 std::vector<int64_t> dilation) {
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(conv_bias.is_contiguous(), "conv_bias must be contiguous");
    TORCH_CHECK(post_bias.is_contiguous(), "post_bias must be contiguous");

    auto output = torch::empty({input.size(0), weight.size(0),
                                (input.size(2) - 1) * stride[0] - 2 * padding[0] + weight.size(2) + output_padding[0],
                                (input.size(3) - 1) * stride[1] - 2 * padding[1] + weight.size(3) + output_padding[1],
                                (input.size(4) - 1) * stride[2] - 2 * padding[2] + weight.size(4) + output_padding[2]},
                               input.options());

    fused_transposed_conv3d_activation_forward(
        input, weight, conv_bias, post_bias, output,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        dilation[0], dilation[1], dilation[2]
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_transposed_conv3d_activation", &fused_transposed_conv3d_activation,
          "Fused Transposed Conv3D + Activation");
}
"""

# -------------------------------------------------------------------------
# Build the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_transposed_conv3d_activation_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional model used for evaluation
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
    # Flatten bias to a 1-D tensor (required by the kernel)
    bias_flat = bias.view(-1)

    # ----- Optimized fused kernel -----
    return fused_ext.fused_transposed_conv3d_activation(
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
stride = [2, 2, 2]
padding = [1, 1, 1]
output_padding = [1, 1, 1]
bias_shape = (out_channels, 1, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
