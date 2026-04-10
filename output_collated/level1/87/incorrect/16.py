# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_071251/code_4.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv1d_weight', 'conv1d_bias', 'conv1d_stride', 'conv1d_padding', 'conv1d_dilation', 'conv1d_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv1d_weight', 'conv1d_bias']


class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool=False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

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
    # State for conv1d (nn.Conv2d)
    if 'conv1d_weight' in flat_state:
        state_kwargs['conv1d_weight'] = flat_state['conv1d_weight']
    else:
        state_kwargs['conv1d_weight'] = getattr(model.conv1d, 'weight', None)
    if 'conv1d_bias' in flat_state:
        state_kwargs['conv1d_bias'] = flat_state['conv1d_bias']
    else:
        state_kwargs['conv1d_bias'] = getattr(model.conv1d, 'bias', None)
    state_kwargs['conv1d_stride'] = model.conv1d.stride
    state_kwargs['conv1d_padding'] = model.conv1d.padding
    state_kwargs['conv1d_dilation'] = model.conv1d.dilation
    state_kwargs['conv1d_groups'] = model.conv1d.groups
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

# Custom CUDA kernel for convolution
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int OC, int KH, int KW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int OH, int OW) {

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N * OC * OH * OW) return;

    // Decompose index
    int tmp = gid;
    int ow = tmp % OW; tmp /= OW;
    int oh = tmp % OH; tmp /= OH;
    int oc = tmp % OC; tmp /= OC;
    int n  = tmp;

    float sum = 0.0f;
    int ih_start = oh * stride_h - pad_h;
    int iw_start = ow * stride_w - pad_w;

    // Inner loops: optimized for standard convolution computation
    for (int ic = 0; ic < C; ++ic) {
        for (int kh = 0; kh < KH; ++kh) {
            int ih = ih_start + kh;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < KW; ++kw) {
                int iw = iw_start + kw;
                if (iw < 0 || iw >= W) continue;

                float val = input[((n * C + ic) * H + ih) * W + iw];
                float w = weight[((oc * C + ic) * KH + kh) * KW + kw];
                sum += val * w;
            }
        }
    }

    if (bias != nullptr) sum += bias[oc];
    output[gid] = sum;
}

void launch_conv2d(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride, int padding) {
    
    int N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    int OC = weight.size(0), KH = weight.size(2), KW = weight.size(3);
    int OH = (H + 2 * padding - KH) / stride + 1;
    int OW = (W + 2 * padding - KW) / stride + 1;

    int total_threads = N * OC * OH * OW;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;

    conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C, H, W, OC, KH, KW, stride, stride, padding, padding, OH, OW
    );
}
"""

cpp_source = r"""
void launch_conv2d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_conv2d", &launch_conv2d, "Run custom conv2d kernel");
}
"""

conv_ext = load_inline(
    name='conv2d_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv1d_weight, conv1d_bias, conv1d_stride, conv1d_padding, conv1d_dilation, conv1d_groups):
    # Note: Assumes 2D convolution logic per requirements
    batch, in_c, h, w = x.shape
    out_c, _, kh, kw = conv1d_weight.shape
    oh = (h + 2 * conv1d_padding - kh) // conv1d_stride + 1
    ow = (w + 2 * conv1d_padding - kw) // conv1d_stride + 1
    
    output = torch.empty((batch, out_c, oh, ow), device=x.device, dtype=x.dtype)
    conv_ext.launch_conv2d(x, conv1d_weight, conv1d_bias, output, conv1d_stride, conv1d_padding)
    return output
