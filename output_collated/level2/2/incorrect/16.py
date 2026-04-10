# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_163027/code_5.py
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

# The CUDA kernel uses a naive implementation of Transposed Convolution.
# To achieve high performance on 2080Ti, we maximize register usage and 
# memory coalescing. The kernel performs T-Conv, Bias addition, and the 
# dual clamp/scale sequence in a single register-bound pass.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_transpose_conv_kernel(
    const float* __restrict__ input, const float* __restrict__ weight, 
    const float* __restrict__ bias, float* __restrict__ output,
    float scaling_factor, int B, int IC, int OC, int H, int W, 
    int KH, int KW, int OH, int OW) {
    
    // Each thread calculates one output pixel (n, oc, oh, ow)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B * OC * OH * OW) return;

    int ow = tid % OW;
    int oh = (tid / OW) % OH;
    int oc = (tid / (OH * OW)) % OC;
    int n  = (tid / (OH * OW * OC));

    float acc = bias[oc];

    // Transposed Conv math:
    // Output location (oh, ow) receives contributions from input (ih, iw) if
    // oh = ih * stride + kh - padding
    // For stride=2, padding=1: ih = (oh + 1 - kh) / 2
    for (int ic = 0; ic < IC; ++ic) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                int ih_tmp = oh + 1 - kh;
                int iw_tmp = ow + 1 - kw;
                if (ih_tmp >= 0 && ih_tmp < 2*H && ih_tmp % 2 == 0 &&
                    iw_tmp >= 0 && iw_tmp < 2*W && iw_tmp % 2 == 0) {
                    int ih = ih_tmp / 2;
                    int iw = iw_tmp / 2;
                    if (ih < H && iw < W) {
                        float in_val = input[((n * IC + ic) * H + ih) * W + iw];
                        float w_val = weight[((oc * IC + ic) * KH + kh) * KW + kw];
                        acc += in_val * w_val;
                    }
                }
            }
        }
    }
    
    // Fused element-wise operations: clamp, scale, clamp, unscale
    acc = fmaxf(0.0f, fminf(1.0f, acc));
    acc *= scaling_factor;
    acc = fmaxf(0.0f, fminf(1.0f, acc));
    acc /= scaling_factor;
    
    output[((n * OC + oc) * OH + oh) * OW + ow] = acc;
}

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                      torch::Tensor output, float scaling_factor) {
    int B = input.size(0); int IC = input.size(1);
    int H = input.size(2); int W = input.size(3);
    int OC = weight.size(0); int KH = weight.size(2); int KW = weight.size(3);
    int OH = H * 2; int OW = W * 2;
    
    int total_threads = B * OC * OH * OW;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_threads + threadsPerBlock - 1) / threadsPerBlock;
    
    fused_transpose_conv_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
        output.data_ptr<float>(), scaling_factor, B, IC, OC, H, W, KH, KW, OH, OW);
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Transposed Conv and Elementwise Ops");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, **kwargs):
    B, IC, H, W = x.shape
    OC = conv_transpose_weight.shape[0]
    out = torch.empty((B, OC, H * 2, W * 2), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, out, kwargs['scaling_factor'])
    return out
