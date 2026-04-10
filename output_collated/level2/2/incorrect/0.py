# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_161645/code_6.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape', 'scaling_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias', 'scaling_factor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

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
    if 'scaling_factor' in flat_state:
        state_kwargs['scaling_factor'] = flat_state['scaling_factor']
    else:
        state_kwargs['scaling_factor'] = getattr(model, 'scaling_factor')
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

# CUDA Kernel: Computes transposed convolution tile-by-tile
# Mapping: Each thread computes the output for one (N, out_c, h, w) coordinate
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_transpose_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kH, int kW,
    int stride, int pad, int out_pad, int dilation,
    float scaling_factor) 
{
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int n_c = blockIdx.z;
    int n = n_c / C_out;
    int oc = n_c % C_out;

    if (w_out >= W_out || h_out >= H_out) return;

    float acc = 0.0f;
    // Transposed convolution accumulation logic
    for (int ic = 0; ic < C_in; ++ic) {
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                int h_in = (h_out + pad - kh * dilation);
                int w_in = (w_out + pad - kw * dilation);
                
                if (h_in % stride == 0 && w_in % stride == 0) {
                    h_in /= stride;
                    w_in /= stride;
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        float val = input[((n * C_in + ic) * H_in + h_in) * W_in + w_in];
                        float w = weight[((ic * C_out + oc) * kH + kh) * kW + kw];
                        acc += val * w;
                    }
                }
            }
        }
    }

    acc += bias[oc];
    acc = fminf(fmaxf(acc, 0.0f), 1.0f);
    acc *= scaling_factor;
    acc = fminf(fmaxf(acc, 0.0f), 1.0f);
    acc /= scaling_factor;

    output[((n * C_out + oc) * H_out + h_out) * W_out + w_out] = acc;
}

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                      torch::Tensor output, float scaling_factor, 
                      int stride, int padding, int out_pad, int groups, int dilation) {
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    const int C_out = output.size(1);
    const int H_out = output.size(2);
    const int W_out = output.size(3);
    const int kH = weight.size(2);
    const int kW = weight.size(3);

    dim3 threads(16, 16);
    dim3 blocks((W_out + 15) / 16, (H_out + 15) / 16, N * C_out);

    fused_transpose_conv_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
        output.data_ptr<float>(), N, C_in, H_in, W_in, C_out, H_out, W_out,
        kH, kW, stride, padding, out_pad, dilation, scaling_factor);
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                      torch::Tensor output, float scaling_factor, 
                      int stride, int padding, int out_pad, int groups, int dilation);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Transposed Conv + Elementwise");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                    conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                    conv_transpose_dilation, bias, scaling_factor):
    N, C_in, H_in, W_in = x.shape
    C_out = conv_transpose_weight.shape[1] * conv_transpose_groups
    H_out = (H_in - 1) * conv_transpose_stride + conv_transpose_output_padding - 2 * conv_transpose_padding + conv_transpose_dilation * (conv_transpose_weight.shape[2] - 1) + 1
    W_out = (W_in - 1) * conv_transpose_stride + conv_transpose_output_padding - 2 * conv_transpose_padding + conv_transpose_dilation * (conv_transpose_weight.shape[3] - 1) + 1
    
    out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, conv_transpose_weight, bias, out, scaling_factor, 
                       conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding, 
                       conv_transpose_groups, conv_transpose_dilation)
    return out
