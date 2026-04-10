# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_120759/code_5.py
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

# The CUDA kernel performs a Transposed Convolution by iterating over the input 
# and accumulating contributions to the output, followed by fusion of clamp and division.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fused_conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int K, int stride, int padding, float min_val, float div) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * D_out * H_out * W_out;
    if (tid >= total_elements) return;

    int n_out = tid / (C_out * D_out * H_out * W_out);
    int rem = tid % (C_out * D_out * H_out * W_out);
    int c_out = rem / (D_out * H_out * W_out);
    rem %= (D_out * H_out * W_out);
    int d_out = rem / (H_out * W_out);
    rem %= (H_out * W_out);
    int h_out = rem / W_out;
    int w_out = rem % W_out;

    float val = bias[c_out];

    // Compute transposed convolution: standard loop for accumulation
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kd = 0; kd < K; ++kd) {
            int d_in_start = d_out + padding - kd;
            if (d_in_start < 0 || d_in_start % stride != 0) continue;
            int d_in = d_in_start / stride;
            if (d_in >= D_in) continue;

            for (int kh = 0; kh < K; ++kh) {
                int h_in_start = h_out + padding - kh;
                if (h_in_start < 0 || h_in_start % stride != 0) continue;
                int h_in = h_in_start / stride;
                if (h_in >= H_in) continue;

                for (int kw = 0; kw < K; ++kw) {
                    int w_in_start = w_out + padding - kw;
                    if (w_in_start < 0 || w_in_start % stride != 0) continue;
                    int w_in = w_in_start / stride;
                    if (w_in >= W_in) continue;

                    int in_idx = (((n_out * C_in + c_in) * D_in + d_in) * H_in + h_in) * W_in + w_in;
                    int w_idx = (((c_in * C_out + c_out) * K + kd) * K + kh) * K + kw;
                    val += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    output[tid] = fmaxf(val, min_val) / div;
}

void fused_op(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
              torch::Tensor output, int stride, int padding, float min_val, float div) {
    int total = output.numel();
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    fused_conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(0), input.size(1), weight.size(1),
        input.size(2), input.size(3), input.size(4),
        output.size(2), output.size(3), output.size(4),
        weight.size(2), stride, padding, min_val, div
    );
}
"""

cpp_source = r"""
void fused_op(torch::Tensor i, torch::Tensor w, torch::Tensor b, torch::Tensor o, int s, int p, float m, float d);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused ConvTranspose3d + Clamp + Div");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, min_value, divisor):
    # Calculate output dimensions
    K = conv_transpose_weight.size(2)
    n = x.size(0)
    c_out = conv_transpose_weight.size(1)
    d_out = (x.size(2) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + (K - 1) * conv_transpose_dilation + 1 + conv_transpose_output_padding
    h_out = (x.size(3) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + (K - 1) * conv_transpose_dilation + 1 + conv_transpose_output_padding
    w_out = (x.size(4) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + (K - 1) * conv_transpose_dilation + 1 + conv_transpose_output_padding
    
    output = torch.empty((n, c_out, d_out, h_out, w_out), device=x.device)
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, output, 
                       conv_transpose_stride, conv_transpose_padding, min_value, divisor)
    return output
