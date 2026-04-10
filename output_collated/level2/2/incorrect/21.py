# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_163844/code_5.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float scaling_factor,
    int N, int IC, int OC, int IH, int IW, int K) {

    int OH = IH * 2;
    int OW = IW * 2;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= N * OC * OH * OW) return;

    int n = out_idx / (OC * OH * OW);
    int oc = (out_idx / (OH * OW)) % OC;
    int oh = (out_idx / OW) % OH;
    int ow = out_idx % OW;

    float val = bias[oc];

    // Direct Deconv (Upsampling 2x with 3x3 kernel)
    // Map output (oh, ow) to input (ih, iw) and evaluate kernel influence
    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            int ih = (oh + 1 - ky);
            int iw = (ow + 1 - kx);
            if (ih >= 0 && ih < IH * 2 && iw >= 0 && iw < IW * 2 && ih % 2 == 0 && iw % 2 == 0) {
                int ih_idx = ih / 2;
                int iw_idx = iw / 2;
                for (int ic = 0; ic < IC; ++ic) {
                    val += input[((n * IC + ic) * IH + ih_idx) * IW + iw_idx] * 
                           weight[((oc * IC + ic) * K + ky) * K + kx];
                }
            }
        }
    }

    // Fused point-wise chain
    val = fminf(fmaxf(val, 0.0f), 1.0f);
    val = fminf(fmaxf(val * scaling_factor, 0.0f), 1.0f) / scaling_factor;
    output[out_idx] = val;
}

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, float scaling_factor) {
    int N = input.size(0); int IC = input.size(1); int IH = input.size(2); int IW = input.size(3);
    int OC = weight.size(1);
    int K = weight.size(2);
    int total_threads = N * OC * (IH * 2) * (IW * 2);
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;
    fused_conv_transpose_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), scaling_factor, N, IC, OC, IH, IW, K);
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, float scaling_factor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused deconv and post-processing");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias, scaling_factor):
    N, C, H, W = x.shape
    OC = conv_transpose_weight.shape[1]
    out = torch.empty((N, OC, H * 2, W * 2), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias.flatten(), out, scaling_factor)
    return out
