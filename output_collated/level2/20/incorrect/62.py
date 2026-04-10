# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_134740/code_9.py
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
import math
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------------------------
# Optimized CUDA kernel – fused transposed convolution + post-processing
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define ELEMENTS_PER_THREAD 4

__global__ void fused_transposed_conv3d_post_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ conv_bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_depth, const int in_height, const int in_width,
    const int out_depth, const int out_height, const int out_width,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int padding_d, const int padding_h, const int padding_w,
    const int output_padding_d, const int output_padding_h, const int output_padding_w,
    const int dilation_d, const int dilation_h, const int dilation_w
) {
    // Each thread processes ELEMENTS_PER_THREAD output elements
    int output_spatial_size = out_depth * out_height * out_width;
    int total_output_elements = batch_size * out_channels * output_spatial_size;
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;

    if (tid >= total_output_elements) return;

    // Precompute channel info and spatial size for post-processing
    const int spatial_size = output_spatial_size;

    // Process ELEMENTS_PER_THREAD elements per thread
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        int idx = tid + i;
        if (idx >= total_output_elements) break;

        // Compute indices
        const int oc = (idx / spatial_size) % out_channels;
        const int batch_idx = idx / (out_channels * spatial_size);
        const int spatial_idx = idx % spatial_size;
        
        const int od = spatial_idx / (out_height * out_width);
        const int oh = (spatial_idx / out_width) % out_height;
        const int ow = spatial_idx % out_width;

        // Load bias for post-processing (using read-only cache)
        float bias_val = __ldg(&bias[oc]);

        // Perform transposed convolution computation
        float conv_sum = 0.0f;
        if (conv_bias != nullptr) {
            conv_sum = __ldg(&conv_bias[oc]);
        }

        // Iterate through kernel dimensions
        for (int kd = 0; kd < kernel_d; ++kd) {
            const int id = od + padding_d - kd * dilation_d;
            if (id % stride_d != 0) continue;
            const int in_d = id / stride_d;
            if (in_d < 0 || in_d >= in_depth) continue;

            for (int kh = 0; kh < kernel_h; ++kh) {
                const int ih = oh + padding_h - kh * dilation_h;
                if (ih % stride_h != 0) continue;
                const int in_h = ih / stride_h;
                if (in_h < 0 || in_h >= in_height) continue;

                for (int kw = 0; kw < kernel_w; ++kw) {
                    const int iw = ow + padding_w - kw * dilation_w;
                    if (iw % stride_w != 0) continue;
                    const int in_w = iw / stride_w;
                    if (in_w < 0 || in_w >= in_width) continue;

                    // Calculate input and weight indices
                    for (int ic = 0; ic < in_channels; ++ic) {
                        const int input_idx = batch_idx * (in_channels * in_depth * in_height * in_width) +
                                              ic * (in_depth * in_height * in_width) +
                                              in_d * (in_height * in_width) +
                                              in_h * in_width + in_w;
                        
                        const int weight_idx = oc * (in_channels * kernel_d * kernel_h * kernel_w) +
                                               ic * (kernel_d * kernel_h * kernel_w) +
                                               kd * (kernel_h * kernel_w) +
                                               kh * kernel_w + kw;
                        
                        conv_sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                    }
                }
            }
        }

        // Apply fused post-processing: ((x + bias) + x) * x + x
        float x = conv_sum;
        float tmp = x + bias_val;          // x + b
        float tmp2 = tmp + x;              // (x + b) + x = 2*x + b
        float res = tmp2 * x + x;          // ((2*x + b) * x) + x

        // Store result
        output[idx] = res;
    }
}

void fused_transposed_conv3d_post_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const c10::optional<torch::Tensor>& conv_bias_opt,
    torch::Tensor& output,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& output_padding,
    const std::vector<int64_t>& dilation
) {
    const int batch_size = static_cast<int>(input.size(0));
    const int in_channels = static_cast<int>(input.size(1));
    const int in_depth = static_cast<int>(input.size(2));
    const int in_height = static_cast<int>(input.size(3));
    const int in_width = static_cast<int>(input.size(4));
    
    const int out_channels = static_cast<int>(weight.size(1));
    const int kernel_d = static_cast<int>(weight.size(2));
    const int kernel_h = static_cast<int>(weight.size(3));
    const int kernel_w = static_cast<int>(weight.size(4));
    
    const int stride_d = static_cast<int>(stride[0]);
    const int stride_h = static_cast<int>(stride[1]);
    const int stride_w = static_cast<int>(stride[2]);
    
    const int padding_d = static_cast<int>(padding[0]);
    const int padding_h = static_cast<int>(padding[1]);
    const int padding_w = static_cast<int>(padding[2]);
    
    const int output_padding_d = static_cast<int>(output_padding[0]);
    const int output_padding_h = static_cast<int>(output_padding[1]);
    const int output_padding_w = static_cast<int>(output_padding[2]);
    
    const int dilation_d = static_cast<int>(dilation[0]);
    const int dilation_h = static_cast<int>(dilation[1]);
    const int dilation_w = static_cast<int>(dilation[2]);
    
    // Calculate output dimensions
    const int out_depth = (in_depth - 1) * stride_d - 2 * padding_d + dilation_d * (kernel_d - 1) + output_padding_d + 1;
    const int out_height = (in_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + output_padding_h + 1;
    const int out_width = (in_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + output_padding_w + 1;
    
    const int total_output_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    // Launch configuration
    const int threads_per_block = 256;
    const int elements_per_thread = ELEMENTS_PER_THREAD;
    const int blocks = (total_output_elements + threads_per_block * elements_per_thread - 1) / (threads_per_block * elements_per_thread);
    
    const at::OptionalTensorRef conv_bias_tensor = conv_bias_opt.has_value() ? 
        at::OptionalTensorRef(conv_bias_opt.value()) : at::OptionalTensorRef();
        
    const float* conv_bias_ptr = conv_bias_opt.has_value() ? 
        conv_bias_opt.value().data_ptr<float>() : nullptr;

    fused_transposed_conv3d_post_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        conv_bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        dilation_d, dilation_h, dilation_w
    );
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>

void fused_transposed_conv3d_post_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const c10::optional<torch::Tensor>& conv_bias,
    torch::Tensor& output,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& output_padding,
    const std::vector<int64_t>& dilation
);

torch::Tensor fused_transposed_conv3d_post(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const c10::optional<torch::Tensor>& conv_bias,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& output_padding,
    const std::vector<int64_t>& dilation
) {
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    
    // Calculate output dimensions
    int64_t N = input.size(0);
    int64_t Co = weight.size(1);
    int64_t Kd = weight.size(2);
    int64_t Kh = weight.size(3);
    int64_t Kw = weight.size(4);
    
    int64_t Id = input.size(2);
    int64_t Ih = input.size(3);
    int64_t Iw = input.size(4);
    
    int64_t Sd = stride[0];
    int64_t Sh = stride[1];
    int64_t Sw = stride[2];
    
    int64_t Pd = padding[0];
    int64_t Ph = padding[1];
    int64_t Pw = padding[2];
    
    int64_t OPd = output_padding[0];
    int64_t OPh = output_padding[1];
    int64_t OPw = output_padding[2];
    
    int64_t Dd = dilation[0];
    int64_t Dh = dilation[1];
    int64_t Dw = dilation[2];
    
    int64_t Od = (Id - 1) * Sd - 2 * Pd + Dd * (Kd - 1) + OPd + 1;
    int64_t Oh = (Ih - 1) * Sh - 2 * Ph + Dh * (Kh - 1) + OPh + 1;
    int64_t Ow = (Iw - 1) * Sw - 2 * Pw + Dw * (Kw - 1) + OPw + 1;
    
    auto output = torch::empty({N, Co, Od, Oh, Ow}, input.options());
    
    fused_transposed_conv3d_post_forward(
        input, weight, bias, conv_bias, output,
        stride, padding, output_padding, dilation
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_transposed_conv3d_post", &fused_transposed_conv3d_post, "Fused transposed conv3d + post processing");
}
"""

# -------------------------------------------------------------------------
# Build the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_transposed_conv3d_post_ext',
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
    # Validate that groups is 1 (our kernel only supports groups=1)
    if conv_transpose_groups != 1:
        raise ValueError("Custom CUDA kernel only supports conv_transpose_groups=1")
    
    # Flatten bias to a 1-D tensor (required by the kernel)
    bias_flat = bias.view(-1)
    
    # Use our custom fused kernel for transposed conv + post-processing
    return fused_ext.fused_transposed_conv3d_post(
        x,
        conv_transpose_weight,
        bias_flat,
        conv_transpose_bias,
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
