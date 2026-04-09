# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_115905/code_13.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'bias_shape', 'stride', 'padding', 'output_padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

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

# CUDA Kernel: Fully fused Transposed Convolution + Bias Subtraction + Tanh
# Note: Implemented using a direct convolution approach for the fused operation.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_conv_transpose_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ act_bias,
    float* __restrict__ output,
    int N, int IC, int OC, int IH, int IW, int OH, int OW,
    int K, int stride, int padding, int groups
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * OC * OH * OW;
    if (tid >= total) return;

    int n = tid / (OC * OH * OW);
    int oc = (tid / (OH * OW)) % OC;
    int oh = (tid / OW) % OH;
    int ow = tid % OW;

    float sum = (conv_bias != nullptr) ? conv_bias[oc] : 0.0f;
    int group_ic = IC / groups;
    int group_id = oc / (OC / groups);

    for (int ic = 0; ic < group_ic; ++ic) {
        int input_c = group_id * group_ic + ic;
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int ih = oh + padding - kh;
                int iw = ow + padding - kw;
                
                if (ih >= 0 && ih < IH * stride && iw >= 0 && iw < IW * stride && 
                    ih % stride == 0 && iw % stride == 0) {
                    
                    int i_idx = ((n * IC + input_c) * IH + (ih / stride)) * IW + (iw / stride);
                    int w_idx = ((group_id * group_ic + ic) * OC/groups + (oc % (OC/groups))) * K * K + kh * K + kw;
                    
                    sum += input[i_idx] * weight[w_idx];
                }
            }
        }
    }
    output[tid] = tanhf(sum - act_bias[oc]);
}

void launch_fused_conv(
    const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& cb, 
    const torch::Tensor& ab, torch::Tensor& out, int s, int p, int g
) {
    int N = x.size(0), IC = x.size(1), IH = x.size(2), IW = x.size(3);
    int OC = w.size(0), K = w.size(2);
    int OH = out.size(2), OW = out.size(3);
    int total = N * OC * OH * OW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    fused_conv_transpose_tanh_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), w.data_ptr<float>(),
        cb.numel() > 0 ? cb.data_ptr<float>() : nullptr,
        ab.data_ptr<float>(), out.data_ptr<float>(),
        N, IC, OC, IH, IW, OH, OW, K, s, p, g
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_fused_conv(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& cb, const torch::Tensor& ab, torch::Tensor& out, int s, int p, int g);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_fused_conv, "Fused Transposed Conv + Bias + Tanh");
}
"""

fused_ext = load_inline(
    name='fused_ext', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias):
    N, IC, IH, IW = x.shape
    OC, _, K, _ = conv_transpose_weight.shape
    OH = (IH - 1) * conv_transpose_stride - 2 * conv_transpose_padding + K + conv_transpose_output_padding
    OW = (IW - 1) * conv_transpose_stride - 2 * conv_transpose_padding + K + conv_transpose_output_padding
    
    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(
        x, conv_transpose_weight, 
        conv_transpose_bias if conv_transpose_bias is not None else torch.tensor([], device=x.device),
        bias.view(-1), out, conv_transpose_stride, conv_transpose_padding, conv_transpose_groups
    )
    return out
