# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_164831/code_7.py
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

# -------------------------------------------------------------------------
# CUDA Kernel: Fused Transposed Conv + Additive Bias + Clamp/Scale Logic
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int CI, const int CO, 
    const int IH, const int IW, 
    const int OH, const int OW,
    const int K, const int S, const int P,
    const float bound)
{
    // Output: [N, CO, OH, OW]
    const int total_out = N * CO * OH * OW;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_out) return;

    // Decoding linear index
    int idx = tid;
    const int ow = idx % OW; idx /= OW;
    const int oh = idx % OH; idx /= OH;
    const int co = idx % CO; idx /= CO;
    const int n  = idx;

    float acc = (conv_bias != nullptr) ? conv_bias[co] : 0.0f;
    acc += bias[co];

    // Transposed conv gather logic
    // For each output (oh, ow), accumulate contributions from input pixels
    for (int ci = 0; ci < CI; ++ci) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                // Relate output coords to input space
                // Input pixel must result in current (oh, ow)
                int ih_virt = oh + P - kh;
                int iw_virt = ow + P - kw;

                if (ih_virt % S == 0 && iw_virt % S == 0) {
                    int ih = ih_virt / S;
                    int iw = iw_virt / S;
                    if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                        float in_val = input[((n * CI + ci) * IH + ih) * IW + iw];
                        float w = weight[((ci * CO + co) * K + kh) * K + kw];
                        acc += in_val * w;
                    }
                }
            }
        }
    }

    // Fused: clamp(x, 0, 1) -> * scale -> clamp(x, 0, 1) -> / scale
    // Mathematically equivalent to: clamp(x, 0.0, 1.0/scaling_factor)
    float res = fmaxf(fminf(acc, bound), 0.0f);
    output[tid] = res;
}

void fused_op_dispatch(
    torch::Tensor input, torch::Tensor weight, torch::Tensor conv_bias,
    torch::Tensor bias, torch::Tensor output, float bound,
    int N, int CI, int CO, int IH, int IW, int OH, int OW, int K, int S, int P)
{
    const int total_elems = N * CO * OH * OW;
    const int threads = 256;
    const int blocks = (total_elems + threads - 1) / threads;
    
    const float* cb_ptr = (conv_bias.numel() > 0) ? conv_bias.data_ptr<float>() : nullptr;
    
    conv_transpose_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), cb_ptr,
        bias.data_ptr<float>(), output.data_ptr<float>(),
        N, CI, CO, IH, IW, OH, OW, K, S, P, bound
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_dispatch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, int, int, int, int, int, int, int, int, int, int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_dispatch, "Fused ConvTranspose Op");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride,
    conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups,
    conv_transpose_dilation, bias, scaling_factor,
):
    N, CI, IH, IW = x.shape
    CO, _, K, _ = conv_transpose_weight.shape
    S, P = conv_transpose_stride, conv_transpose_padding
    
    # Calculate output dimensions
    OH = (IH - 1) * S - 2 * P + K + conv_transpose_output_padding
    OW = (IW - 1) * S - 2 * P + K + conv_transpose_output_padding
    
    out = torch.empty((N, CO, OH, OW), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(
        x.contiguous(), conv_transpose_weight.contiguous(), 
        conv_transpose_bias if conv_transpose_bias is not None else torch.tensor([]),
        bias.contiguous(), out, 1.0 / scaling_factor,
        N, CI, CO, IH, IW, OH, OW, K, S, P
    )
    return out
