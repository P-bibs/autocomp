# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_051208/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies HardSwish, and then ReLU.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

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

# --- CUDA Kernel ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_act_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int ic, int oc, int ih, int iw, int k,
    int stride, int pad, int dilation, int oh, int ow
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * oc * oh * ow;
    if (idx >= total) return;

    int n = idx / (oc * oh * ow);
    int rem = idx % (oc * oh * ow);
    int o = rem / (oh * ow);
    int h_out = (rem / ow) % oh;
    int w_out = rem % ow;

    float val = (bias != nullptr) ? bias[o] : 0.0f;

    for (int c = 0; c < ic; ++c) {
        for (int kh = 0; kh < k; ++kh) {
            int h_in = h_out * stride + kh * dilation - pad;
            if (h_in < 0 || h_in >= ih) continue;
            for (int kw = 0; kw < k; ++kw) {
                int w_in = w_out * stride + kw * dilation - pad;
                if (w_in < 0 || w_in >= iw) continue;
                
                float i_v = __ldg(&input[((n * ic + c) * ih + h_in) * iw + w_in]);
                float w_v = __ldg(&weight[((o * ic + c) * k + kh) * k + kw]);
                val += i_v * w_v;
            }
        }
    }

    // HardSwish: x * relu6(x + 3) / 6
    float hs = val * fminf(fmaxf(val + 3.0f, 0.0f), 6.0f) * 0.16666667f;
    // ReLU
    output[idx] = fmaxf(hs, 0.0f);
}

void fused_op_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride, int padding, int dilation, int groups
) {
    int b = input.size(0), ic = input.size(1), ih = input.size(2), iw = input.size(3);
    int oc = weight.size(0), k = weight.size(2);
    int oh = (ih + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    int ow = (iw + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    
    int total = b * oc * oh * ow;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    fused_conv_act_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr, output.data_ptr<float>(),
        b, ic, oc, ih, iw, k, stride, padding, dilation, oh, ow
    );
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int padding, int dilation, int groups);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Forward Op");
}
"""

fused_ext = load_inline(name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_kernel, 
                       extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    kh = conv_weight.size(2)
    oh = (x.size(2) + 2 * conv_padding - conv_dilation * (kh - 1) - 1) // conv_stride + 1
    ow = (x.size(3) + 2 * conv_padding - conv_dilation * (kh - 1) - 1) // conv_stride + 1
    out = torch.empty(x.size(0), conv_weight.size(0), oh, ow, device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, conv_weight, conv_bias, out, conv_stride, conv_padding, conv_dilation, conv_groups)
    return out
