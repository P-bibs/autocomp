# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_113742/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'min_value', 'divisor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'min_value', 'divisor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

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
    if 'min_value' in flat_state:
        state_kwargs['min_value'] = flat_state['min_value']
    else:
        state_kwargs['min_value'] = getattr(model, 'min_value')
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
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

# The CUDA source code implements a manual gather-based 3D transposed convolution.
# We fuse the accumulation, clamp, and division steps to maximize memory bandwidth utilization.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void conv_transpose_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int stride, const int padding,
    const int op_pad, const int dilation,
    const float min_value,
    const float divisor)
{
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_out = N * C_out * D_out * H_out * W_out;
    if (out_idx >= total_out) return;

    // Coordinate decomposition
    int idx = out_idx;
    const int ow = idx % W_out; idx /= W_out;
    const int oh = idx % H_out; idx /= H_out;
    const int od = idx % D_out; idx /= D_out;
    const int oc = idx % C_out; idx /= C_out;
    const int n  = idx;

    float sum = bias[oc];
    constexpr int K = 3;

    // Gather-style transposed convolution
    // We iterate over input channels and kernel positions that contribute to the current output pixel
    for (int ic = 0; ic < C_in; ++ic) {
        const float* w_ptr = weight + ((ic * C_out + oc) * K * K * K);
        
        for (int kd = 0; kd < K; ++kd) {
            int id = (od + padding - kd * dilation - op_pad);
            if (id < 0 || id % stride != 0) continue;
            id /= stride;
            if (id >= D_in) continue;

            for (int kh = 0; kh < K; ++kh) {
                int ih = (oh + padding - kh * dilation - op_pad);
                if (ih < 0 || ih % stride != 0) continue;
                ih /= stride;
                if (ih >= H_in) continue;

                for (int kw = 0; kw < K; ++kw) {
                    int iw = (ow + padding - kw * dilation - op_pad);
                    if (iw < 0 || iw % stride != 0) continue;
                    iw /= stride;
                    if (iw >= W_in) continue;

                    int in_idx = (((n * C_in + ic) * D_in + id) * H_in + ih) * W_in + iw;
                    sum += input[in_idx] * w_ptr[(kd * K * K + kh * K + kw)];
                }
            }
        }
    }

    output[out_idx] = fmaxf(sum, min_value) / divisor;
}

void launch_fused_op(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, torch::Tensor& output,
    int stride, int padding, int op_pad, int dilation, float min_value, float divisor)
{
    const int N = input.size(0), C_in = input.size(1), D_in = input.size(2), H_in = input.size(3), W_in = input.size(4);
    const int C_out = weight.size(1), D_out = output.size(2), H_out = output.size(3), W_out = output.size(4);
    
    int total_elements = N * C_out * D_out * H_out * W_out;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    conv_transpose_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        N, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out,
        stride, padding, op_pad, dilation, min_value, divisor);
}
"""

cpp_source = r"""
void launch_fused_op(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, torch::Tensor&, int, int, int, int, float, float);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_fused_op, "Fused ConvTranspose3d + Clamp + Div");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, min_value, divisor):
    # Weight shape: (C_in, C_out, K, K, K)
    N, C_in, D_in, H_in, W_in = x.shape
    C_out, K = conv_transpose_weight.shape[1], conv_transpose_weight.shape[2]
    
    D_out = (D_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1
    H_out = (H_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1
    W_out = (W_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1
    
    out = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    bias = conv_transpose_bias if conv_transpose_bias is not None else torch.zeros(C_out, device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(x, conv_transpose_weight, bias, out, conv_transpose_stride, 
                       conv_transpose_padding, conv_transpose_output_padding, conv_transpose_dilation, 
                       float(min_value), float(divisor))
    return out
