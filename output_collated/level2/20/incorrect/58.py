# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_134740/code_4.py
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

# Fully fused ConvTranspose3d + bias addition + elementwise arithmetic kernel
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

// Helper to compute 3D index from linear index
__device__ __forceinline__ void get_5d_index(
    int idx, int D, int H, int W, int C,
    int& batch, int& c, int& d, int& h, int& w) {
    w = idx % W; idx /= W;
    h = idx % H; idx /= H;
    d = idx % D; idx /= D;
    c = idx % C; idx /= C;
    batch = idx;
}

__global__ void fused_conv_transpose_arithmetic_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ post_bias,
    float* __restrict__ output,
    int batch, int in_c, int out_c,
    int in_D, int in_H, int in_W,
    int kD, int kH, int kW,
    int strideD, int strideH, int strideW,
    int paddingD, int paddingH, int paddingW,
    int output_paddingD, int output_paddingH, int output_paddingW,
    int out_D, int out_H, int out_W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * out_c * out_D * out_H * out_W;

    if (idx >= total_elements) return;

    // Decompose linear index into 5D coordinates
    int batch_id, out_c_id, out_d, out_h, out_w;
    get_5d_index(idx, out_D, out_H, out_W, out_c, batch_id, out_c_id, out_d, out_h, out_w);

    // Calculate input coordinates that contribute to this output point via transposed convolution
    float accumulator = 0.0f;

    // Loop over kernel dimensions
    for (int kd = 0; kd < kD; kd++) {
        int in_d_calc = out_d + paddingD - kd;
        if (in_d_calc % strideD != 0) continue;
        int in_d = in_d_calc / strideD;
        if (in_d < 0 || in_d >= in_D) continue;

        for (int kh = 0; kh < kH; kh++) {
            int in_h_calc = out_h + paddingH - kh;
            if (in_h_calc % strideH != 0) continue;
            int in_h = in_h_calc / strideH;
            if (in_h < 0 || in_h >= in_H) continue;

            for (int kw = 0; kw < kW; kw++) {
                int in_w_calc = out_w + paddingW - kw;
                if (in_w_calc % strideW != 0) continue;
                int in_w = in_w_calc / strideW;
                if (in_w < 0 || in_w >= in_W) continue;

                // Iterate through input channels (assuming groups=1)
                for (int in_ch = 0; in_ch < in_c; ++in_ch) {
                    int weight_idx = out_c_id * (in_c * kD * kH * kW) +
                                     in_ch * (kD * kH * kW) +
                                     kd * (kH * kW) +
                                     kh * kW +
                                     kw;

                    int input_idx = batch_id * (in_c * in_D * in_H * in_W) +
                                    in_ch * (in_D * in_H * in_W) +
                                    in_d * (in_H * in_W) +
                                    in_h * in_W +
                                    in_w;

                    accumulator += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add convolution bias
    accumulator += conv_bias[out_c_id];

    // Apply element-wise arithmetic: ((x + b) + x) * x + x = (2*x + b) * x + x
    float b = post_bias[out_c_id];
    float result = ((accumulator + b) + accumulator) * accumulator + accumulator;

    output[idx] = result;
}

void fused_conv_transpose_forward(
    const torch::Tensor& input,    // [N, C_in, D_in, H_in, W_in]
    const torch::Tensor& weight,   // [C_in, C_out/groups, kD, kH, kW]
    const torch::Tensor& conv_bias, // [C_out]
    const torch::Tensor& post_bias, // [C_out]
    torch::Tensor& output,         // [N, C_out, D_out, H_out, W_out]
    int strideD, int strideH, int strideW,
    int paddingD, int paddingH, int paddingW,
    int output_paddingD, int output_paddingH, int output_paddingW
) {
    int batch = input.size(0);
    int in_c = input.size(1);
    int in_D = input.size(2);
    int in_H = input.size(3);
    int in_W = input.size(4);

    int out_c = weight.size(1); // C_out/groups, assuming groups=1
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    // Compute output dimensions
    int out_D = (in_D - 1) * strideD - 2 * paddingD + kD + output_paddingD;
    int out_H = (in_H - 1) * strideH - 2 * paddingH + kH + output_paddingH;
    int out_W = (in_W - 1) * strideW - 2 * paddingW + kW + output_paddingW;

    int total_threads = batch * out_c * out_D * out_H * out_W;
    int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    fused_conv_transpose_arithmetic_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        post_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, in_c, out_c,
        in_D, in_H, in_W,
        kD, kH, kW,
        strideD, strideH, strideW,
        paddingD, paddingH, paddingW,
        output_paddingD, output_paddingH, output_paddingW,
        out_D, out_H, out_W
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_forward(
    const torch::Tensor& input, const torch::Tensor& weight,
    const torch::Tensor& conv_bias, const torch::Tensor& post_bias,
    torch::Tensor& output,
    int strideD, int strideH, int strideW,
    int paddingD, int paddingH, int paddingW,
    int output_paddingD, int output_paddingH, int output_paddingW
);

void fused_conv_transpose_wrapper(
    const torch::Tensor& input, const torch::Tensor& weight,
    const torch::Tensor& conv_bias, const torch::Tensor& post_bias,
    torch::Tensor& output,
    std::vector<int64_t> stride, std::vector<int64_t> padding,
    std::vector<int64_t> output_padding
) {
    fused_conv_transpose_forward(
        input, weight, conv_bias, post_bias, output,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2]
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose", &fused_conv_transpose_wrapper, "Fused ConvTranspose3D with arithmetic");
}
"""

fused_ext = load_inline(
    name='fused_conv_transpose_ext',
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
    # Ensure all parameters are correctly shaped
    if conv_transpose_groups != 1:
        raise NotImplementedError("Only groups=1 is supported in custom kernel")

    if not isinstance(conv_transpose_stride, (tuple, list)):
        conv_transpose_stride = (conv_transpose_stride,) * 3
    if not isinstance(conv_transpose_padding, (tuple, list)):
        conv_transpose_padding = (conv_transpose_padding,) * 3
    if not isinstance(conv_transpose_output_padding, (tuple, list)):
        conv_transpose_output_padding = (conv_transpose_output_padding,) * 3
    if not isinstance(conv_transpose_dilation, (tuple, list)):
        conv_transpose_dilation = (conv_transpose_dilation,) * 3

    # Compute output shape
    in_D, in_H, in_W = x.shape[2:]
    kD, kH, kW = conv_transpose_weight.shape[2:]
    strideD, strideH, strideW = conv_transpose_stride
    paddingD, paddingH, paddingW = conv_transpose_padding
    output_paddingD, output_paddingH, output_paddingW = conv_transpose_output_padding

    out_D = (in_D - 1) * strideD - 2 * paddingD + kD + output_paddingD
    out_H = (in_H - 1) * strideH - 2 * paddingH + kH + output_paddingH
    out_W = (in_W - 1) * strideW - 2 * paddingW + kW + output_paddingW

    out_channels = conv_transpose_weight.shape[1]  # C_out/groups for groups=1
    output_shape = (x.shape[0], out_channels, out_D, out_H, out_W)
    
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    bias_flat = bias.view(-1)

    fused_ext.fused_conv_transpose(
        x.contiguous(),
        conv_transpose_weight.contiguous(),
        conv_transpose_bias.contiguous(),
        bias_flat.contiguous(),
        output,
        list(conv_transpose_stride),
        list(conv_transpose_padding),
        list(conv_transpose_output_padding)
    )

    return output

# Parameters matching original for testing/evaluation consistency
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
