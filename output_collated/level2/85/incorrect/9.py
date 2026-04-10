# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_142418/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'num_groups', 'scale_shape', 'maxpool_kernel_size', 'clamp_min', 'clamp_max']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'group_norm_weight', 'group_norm_bias', 'group_norm_num_groups', 'group_norm_eps', 'maxpool_kernel_size', 'maxpool_stride', 'maxpool_padding', 'maxpool_dilation', 'maxpool_ceil_mode', 'maxpool_return_indices', 'scale', 'clamp_min', 'clamp_max']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'group_norm_weight', 'group_norm_bias', 'scale']


class ModelNew(nn.Module):
    """
    ModelNew that performs convolution, group normalization, scaling, max pooling, and clamping.
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

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
    # State for group_norm (nn.GroupNorm)
    if 'group_norm_weight' in flat_state:
        state_kwargs['group_norm_weight'] = flat_state['group_norm_weight']
    else:
        state_kwargs['group_norm_weight'] = getattr(model.group_norm, 'weight', None)
    if 'group_norm_bias' in flat_state:
        state_kwargs['group_norm_bias'] = flat_state['group_norm_bias']
    else:
        state_kwargs['group_norm_bias'] = getattr(model.group_norm, 'bias', None)
    state_kwargs['group_norm_num_groups'] = model.group_norm.num_groups
    state_kwargs['group_norm_eps'] = model.group_norm.eps
    # State for maxpool (nn.MaxPool2d)
    state_kwargs['maxpool_kernel_size'] = model.maxpool.kernel_size
    state_kwargs['maxpool_stride'] = model.maxpool.stride
    state_kwargs['maxpool_padding'] = model.maxpool.padding
    state_kwargs['maxpool_dilation'] = model.maxpool.dilation
    state_kwargs['maxpool_ceil_mode'] = model.maxpool.ceil_mode
    state_kwargs['maxpool_return_indices'] = model.maxpool.return_indices
    if 'scale' in flat_state:
        state_kwargs['scale'] = flat_state['scale']
    else:
        state_kwargs['scale'] = getattr(model, 'scale')
    if 'clamp_min' in flat_state:
        state_kwargs['clamp_min'] = flat_state['clamp_min']
    else:
        state_kwargs['clamp_min'] = getattr(model, 'clamp_min')
    if 'clamp_max' in flat_state:
        state_kwargs['clamp_max'] = flat_state['clamp_max']
    else:
        state_kwargs['clamp_max'] = getattr(model, 'clamp_max')
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# The CUDA kernel performs a grouped convolution.
# We map each (batch, output_channel, out_h, out_w) to a GPU thread.
conv_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C, const int H, const int W,
    const int outC, const int outH, const int outW,
    const int kernel_size, const int stride, const int padding, const int groups) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * outC * outH * outW;
    if (idx >= total) return;

    int tmp = idx;
    const int ow = tmp % outW; tmp /= outW;
    const int oh = tmp % outH; tmp /= outH;
    const int oc = tmp % outC; tmp /= outC;
    const int n = tmp;

    const int inC_per_group = C / groups;
    const int outC_per_group = outC / groups;
    const int group_id = oc / outC_per_group;
    const int oc_local = oc % outC_per_group;

    float sum = bias[oc];

    const int in_base = n * C * H * W + group_id * inC_per_group * H * W;
    const int wt_base = oc * inC_per_group * kernel_size * kernel_size;

    for (int ic = 0; ic < inC_per_group; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            int ih = oh * stride - padding + kh;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < kernel_size; ++kw) {
                int iw = ow * stride - padding + kw;
                if (iw < 0 || iw >= W) continue;
                
                sum += input[in_base + ic * H * W + ih * W + iw] * 
                       weight[wt_base + ic * kernel_size * kernel_size + kh * kernel_size + kw];
            }
        }
    }
    output[idx] = sum;
}

void conv2d_cuda_impl(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output, int stride, int padding, int groups) {
    
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int outC = output.size(1);
    const int outH = output.size(2);
    const int outW = output.size(3);
    const int kS = weight.size(2);

    const int total = N * outC * outH * outW;
    const int block = 256;
    const int grid = (total + block - 1) / block;

    conv_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, C, H, W, outC, outH, outW, kS, stride, padding, groups);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void conv2d_cuda_impl(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int padding, int groups);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d", &conv2d_cuda_impl, "Custom CUDA grouped conv2d");
}
"""

# Compile the extension
conv_ext = load_inline(
    name='conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=conv_cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups,
    group_norm_weight, group_norm_bias, group_norm_num_groups, group_norm_eps,
    maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation,
    maxpool_ceil_mode, maxpool_return_indices, scale, clamp_min, clamp_max,
):
    # Ensure all inputs are on GPU
    x, conv_weight, conv_bias = x.cuda(), conv_weight.cuda(), conv_bias.cuda()
    
    # Calculate output dims
    kH = conv_weight.size(2)
    outH = (x.size(2) + 2 * conv_padding - conv_dilation * (kH - 1) - 1) // conv_stride + 1
    outW = (x.size(3) + 2 * conv_padding - conv_dilation * (kH - 1) - 1) // conv_stride + 1
    output = torch.empty((x.size(0), conv_weight.size(0), outH, outW), device='cuda')
    
    # Custom kernel execution
    conv_ext.conv2d(x, conv_weight, conv_bias, output, conv_stride, conv_padding, conv_groups)
    
    # Subsequent operations on GPU
    x = F.group_norm(output, group_norm_num_groups, group_norm_weight.cuda(), group_norm_bias.cuda(), group_norm_eps)
    x = x * scale
    x = F.max_pool2d(x, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode)
    return torch.clamp(x, clamp_min, clamp_max)
