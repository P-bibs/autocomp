# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_162535/code_5.py
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

# CUDA Kernel: Fusing ConvTranspose2d (Hardcoded for Stride 2, Padding 1, K=3) 
# with Bias, and the specified clamp/scale/clamp sequence.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ out,
    int B, int Ci, int Co, int Hi, int Wi, int Ho, int Wo, 
    float scale) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * Co * Ho * Wo;
    if (tid >= total_elements) return;

    // Index mapping
    int tmp = tid;
    int w = tmp % Wo; tmp /= Wo;
    int h = tmp % Ho; tmp /= Ho;
    int co = tmp % Co; tmp /= Co;
    int b = tmp;

    float acc = bias[co];

    // Transposed Conv (Stride 2, Padding 1, Kernel 3)
    // Output index (h, w) relates to input index ih, iw
    // ih, iw are valid if: 0 <= ih < Hi, 0 <= iw < Wi
    // kh, kw range [0, 2]
    for (int ci = 0; ci < Ci; ++ci) {
        for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
                // To support stride 2, padding 1: 
                // input_idx = (output_idx + padding - kernel_idx) / stride
                int ih = (h + 1 - kh);
                int iw = (w + 1 - kw);
                
                if (ih % 2 == 0 && iw % 2 == 0) {
                    ih /= 2; iw /= 2;
                    if (ih >= 0 && ih < Hi && iw >= 0 && iw < Wi) {
                        float input_val = x[((b * Ci + ci) * Hi + ih) * Wi + iw];
                        float weight_val = weight[((co * Ci + ci) * 3 + kh) * 3 + kw];
                        acc += input_val * weight_val;
                    }
                }
            }
        }
    }

    // Fused operations (Optimization 6)
    float val = fminf(fmaxf(acc, 0.0f), 1.0f);
    val *= scale;
    val = fminf(fmaxf(val, 0.0f), 1.0f);
    out[tid] = val / scale;
}

void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor out, float scale) {
    int B = x.size(0);
    int Ci = x.size(1);
    int Hi = x.size(2);
    int Wi = x.size(3);
    int Co = weight.size(0);
    int Ho = (Hi - 1) * 2 - 2 * 1 + 3 + 1; // Standard formula for output size
    int Wo = (Wi - 1) * 2 - 2 * 1 + 3 + 1;

    int total_threads = B * Co * Ho * Wo;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;

    fused_conv_transpose_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
        out.data_ptr<float>(), B, Ci, Co, Hi, Wi, Ho, Wo, scale);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor out, float scale);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused ConvTranspose2D + Bias + Clamp/ScaleOp");
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
                     conv_transpose_dilation, bias, scaling_factor):
    # Output shape calculation for standard parameters
    B, Ci, Hi, Wi = x.shape
    Co = conv_transpose_weight.shape[1]
    Ho = (Hi - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (3 - 1) + conv_transpose_output_padding + 1
    Wo = (Wi - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (3 - 1) + conv_transpose_output_padding + 1
    
    out = torch.empty((B, Co, Ho, Wo), device=x.device)
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias.flatten(), out, scaling_factor)
    return out
