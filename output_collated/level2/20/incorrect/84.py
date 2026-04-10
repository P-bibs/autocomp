# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_140117/code_23.py
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

# Optimized CUDA kernel to perform transposed convolution 3D followed by the fused activation
# f(x) = ((x + b) + x) * x + x integrated into the output write stage.
# We implement a basic but vectorized approach for performance.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void fused_conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int C_out, 
    int D, int H, int W,
    int kD, int kH, int kW,
    int outD, int outH, int outW,
    int stride, int padding
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C_out * outD * outH * outW;

    if (tid < total_elements) {
        int temp = tid;
        int w_idx = temp % outW; temp /= outW;
        int h_idx = temp % outH; temp /= outH;
        int d_idx = temp % outD; temp /= outD;
        int co_idx = temp % C_out; temp /= C_out;
        int b_idx = temp;

        float val = 0.0f;
        // Simple 3D Transposed Convolution Logic
        for (int ci = 0; ci < C_in; ++ci) {
            for (int kd = 0; kd < kD; ++kd) {
                for (int kh = 0; kh < kH; ++kh) {
                    for (int kw = 0; kw < kW; ++kw) {
                        int in_d = (d_idx + padding - kd);
                        int in_h = (h_idx + padding - kh);
                        int in_w = (w_idx + padding - kw);
                        
                        if (in_d % stride == 0 && in_h % stride == 0 && in_w % stride == 0) {
                            in_d /= stride; in_h /= stride; in_w /= stride;
                            if (in_d >= 0 && in_d < D && in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                                int input_idx = (((b_idx * C_in + ci) * D + in_d) * H + in_h) * W + in_w;
                                int weight_idx = (((ci * C_out + co_idx) * kD + kd) * kH + kh) * kW + kw;
                                val += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
        }
        
        float b = bias[co_idx];
        // Apply fused arithmetic: ((x + b) + x) * x + x = (2x + b) * x + x = 2x^2 + bx + x
        output[tid] = ((val + b) + val) * val + val;
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_cuda(const torch::Tensor& input, const torch::Tensor& weight, 
                                 const torch::Tensor& bias, torch::Tensor& output, 
                                 int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d", &fused_conv_transpose3d_cuda, "Fused ConvTranspose3D + Activation");
}
"""

cuda_source_full = cuda_kernel + r"""
void fused_conv_transpose3d_cuda(const torch::Tensor& input, const torch::Tensor& weight, 
                                 const torch::Tensor& bias, torch::Tensor& output, 
                                 int stride, int padding) {
    int B = input.size(0); int C_in = input.size(1);
    int D = input.size(2); int H = input.size(3); int W = input.size(4);
    int C_out = weight.size(1);
    int kD = weight.size(2); int kH = weight.size(3); int kW = weight.size(4);
    int outD = output.size(2); int outH = output.size(3); int outW = output.size(4);

    int total_elements = B * C_out * outD * outH * outW;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), B, C_in, C_out, D, H, W, kD, kH, kW, 
        outD, outH, outW, stride, padding
    );
}
"""

fused_ext = load_inline(
    name='fused_conv_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source_full,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, 
                     conv_transpose_stride, conv_transpose_padding, 
                     conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias):
    stride = conv_transpose_stride[0]
    padding = conv_transpose_padding[0]
    
    # Calculate output dimensions
    B, C_in, D, H, W = x.shape
    C_out = conv_transpose_weight.shape[1]
    outD = (D - 1) * stride - 2 * padding + conv_transpose_weight.shape[2] + conv_transpose_output_padding[0]
    outH = (H - 1) * stride - 2 * padding + conv_transpose_weight.shape[3] + conv_transpose_output_padding[1]
    outW = (W - 1) * stride - 2 * padding + conv_transpose_weight.shape[4] + conv_transpose_output_padding[2]
    
    output = torch.empty((B, C_out, outD, outH, outW), device=x.device)
    fused_ext.fused_conv_transpose3d(x.contiguous(), conv_transpose_weight.contiguous(), 
                                      bias.flatten().contiguous(), output, stride, padding)
    return output
