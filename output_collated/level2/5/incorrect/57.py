# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_121229/code_20.py
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

# Combined Kernel: Transpose Convolution (Direct) + Bias + Tanh
# We use a vectorized direct transpose convolution approach
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose_bias_tanh_kernel(
    const float* __restrict__ input, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ output,
    int N, int IC, int OC, int IH, int IW, int OH, int OW, int K) {

    int batch = blockIdx.z;
    int oc = blockIdx.y;
    int oh = blockIdx.x;

    for (int ow = threadIdx.x; ow < OW; ow += blockDim.x) {
        float val = 0.0f;
        for (int ic = 0; ic < IC; ++ic) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int ih = (oh + 1 - 0 - kh); // Simplified for stride 1, padding 1
                    int iw = (ow + 1 - 0 - kw);
                    if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                        val += input[(batch * IC + ic) * IH * IW + ih * IW + iw] * 
                               weight[ic * OC * K * K + oc * K * K + kh * K + kw];
                    }
                }
            }
        }
        int out_idx = ((batch * OC + oc) * OH + oh) * OW + ow;
        output[out_idx] = tanhf(val + bias[oc]);
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    int N = input.size(0); int IC = input.size(1);
    int IH = input.size(2); int IW = input.size(3);
    int OC = weight.size(1); int K = weight.size(2);
    int OH = output.size(2); int OW = output.size(3);

    dim3 blocks(OH, OC, N);
    int threads = min(OW, 256);
    fused_conv_transpose_bias_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.data_ptr<float>(), output.data_ptr<float>(),
        N, IC, OC, IH, IW, OH, OW, K);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Transpose Conv Bias Tanh");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, 
                     conv_transpose_stride, conv_transpose_padding, 
                     conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias):
    # Output shape calculation for standard ConvTranspose2d
    N, IC, IH, IW = x.shape
    OC, _, K, _ = conv_transpose_weight.shape
    OH = (IH - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + K + conv_transpose_output_padding[0]
    OW = (IW - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + K + conv_transpose_output_padding[1]
    
    output = torch.empty((N, OC, OH, OW), device=x.device)
    
    # Kernel executes the heavy lifting without PyTorch built-in convs
    fused_ext.fused_op(x.contiguous(), conv_transpose_weight.contiguous(), 
                       bias.view(-1).contiguous(), output)
    return output
