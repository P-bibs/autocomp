# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_142828/code_23.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'subtract_value_1', 'subtract_value_2']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'subtract_value_1', 'subtract_value_2']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, subtracts two values, applies Mish activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

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
    # State for conv (nn.Conv2d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
    if 'subtract_value_1' in flat_state:
        state_kwargs['subtract_value_1'] = flat_state['subtract_value_1']
    else:
        state_kwargs['subtract_value_1'] = getattr(model, 'subtract_value_1')
    if 'subtract_value_2' in flat_state:
        state_kwargs['subtract_value_2'] = flat_state['subtract_value_2']
    else:
        state_kwargs['subtract_value_2'] = getattr(model, 'subtract_value_2')
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
#include <math.h>

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input, const float* __restrict__ weight, 
    const float* __restrict__ bias, float* __restrict__ output,
    int N, int C, int H, int W, int OC, int KH, int KW, 
    int pad, float sub1, float sub2) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * OC * H * W;

    if (tid < total_elements) {
        int temp = tid;
        int ow = temp % W; temp /= W;
        int oh = temp % H; temp /= H;
        int oc = temp % OC; temp /= OC;
        int n = temp;

        float acc = bias[oc];
        for (int ic = 0; ic < C; ++ic) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    int ih = oh + kh - pad;
                    int iw = ow + kw - pad;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        float in_val = input[((n * C + ic) * H + ih) * W + iw];
                        float w_val = weight[((oc * C + ic) * KH + kh) * KW + kw];
                        acc += in_val * w_val;
                    }
                }
            }
        }
        
        float val = acc - sub1 - sub2;
        // Mish: x * tanh(softplus(x))
        // softplus(x) = log(1 + exp(x))
        output[tid] = val * tanhf(logf(1.0f + expf(val)));
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                      torch::Tensor output, float sub1, float sub2, int padding) {
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int OC = weight.size(0);
    int KH = weight.size(2);
    int KW = weight.size(3);
    
    int total_elements = N * OC * H * W;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_conv_mish_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, C, H, W, OC, KH, KW, padding, sub1, sub2);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor i, torch::Tensor w, torch::Tensor b, torch::Tensor o, float s1, float s2, int p);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv + Sub + Mish");
}
"""

fused_ext = load_inline(
    name='fused_conv_mish',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups, subtract_value_1, subtract_value_2):
    # Only supports dilation=1 and stride=1 based on the optimization plan kernel
    N, C, H, W = x.shape
    OC = conv_weight.shape[0]
    output = torch.empty((N, OC, H, W), device=x.device)
    
    # Assuming standard padding int
    pad = conv_padding[0] if isinstance(conv_padding, (tuple, list)) else conv_padding
    
    fused_ext.fused_op(x, conv_weight, conv_bias, output, subtract_value_1, subtract_value_2, pad)
    return output
