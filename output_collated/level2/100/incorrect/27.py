# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_125214/code_5.py
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

# The CUDA kernel implements Transposed Convolution 3D as a direct gathering operation.
# It computes indices in reverse and fuses bias addition, clamping, and division.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int Ci, int Co, int Di, int Hi, int Wi,
    int KD, int KH, int KW, int S, int P,
    int Do, int Ho, int Wo, float min_val, float divisor) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = B * Co * Do * Ho * Wo;
    
    if (tid >= total_out) return;

    int idx = tid;
    int ow = idx % Wo; idx /= Wo;
    int oh = idx % Ho; idx /= Ho;
    int od = idx % Do; idx /= Do;
    int co = idx % Co; idx /= Co;
    int b  = idx;

    float sum = bias[co];

    for (int ci = 0; ci < Ci; ++ci) {
        for (int kd = 0; kd < KD; ++kd) {
            int id_in = od + P - kd;
            if (id_in < 0 || id_in % S != 0) continue;
            int ii = id_in / S;
            if (ii >= Di) continue;

            for (int kh = 0; kh < KH; ++kh) {
                int ih_in = oh + P - kh;
                if (ih_in < 0 || ih_in % S != 0) continue;
                int ij = ih_in / S;
                if (ij >= Hi) continue;

                for (int kw = 0; kw < KW; ++kw) {
                    int iw_in = ow + P - kw;
                    if (iw_in < 0 || iw_in % S != 0) continue;
                    int ik = iw_in / S;
                    if (ik >= Wi) continue;

                    float val = input[((b * Ci + ci) * Di + ii) * Hi * Wi + ij * Wi + ik];
                    float w = weight[((ci * Co + co) * KD + kd) * KH * KW + kh * KW + kw];
                    sum += val * w;
                }
            }
        }
    }
    output[tid] = fmaxf(sum, min_val) / divisor;
}

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                      torch::Tensor output, int S, int P, float min_val, float divisor) {
    int B = input.size(0); int Ci = input.size(1);
    int Di = input.size(2); int Hi = input.size(3); int Wi = input.size(4);
    int Co = weight.size(1); int KD = weight.size(2); int KH = weight.size(3); int KW = weight.size(4);
    int Do = (Di - 1) * S + KD - 2 * P;
    int Ho = (Hi - 1) * S + KH - 2 * P;
    int Wo = (Wi - 1) * S + KW - 2 * P;
    
    int total = B * Co * Do * Ho * Wo;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    fused_conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
        output.data_ptr<float>(), B, Ci, Co, Di, Hi, Wi, KD, KH, KW, S, P, Do, Ho, Wo, min_val, divisor);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, float, float);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Transposed Conv3D + Clamp + Div");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, min_value, divisor):
    B, Ci, Di, Hi, Wi = x.shape
    _, _, KD, KH, KW = conv_transpose_weight.shape
    Do = (Di - 1) * conv_transpose_stride + KD - 2 * conv_transpose_padding
    Ho = (Hi - 1) * conv_transpose_stride + KH - 2 * conv_transpose_padding
    Wo = (Wi - 1) * conv_transpose_stride + KW - 2 * conv_transpose_padding
    
    output = torch.empty((B, conv_transpose_weight.size(1), Do, Ho, Wo), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, output, 
                       conv_transpose_stride, conv_transpose_padding, min_value, divisor)
    return output
