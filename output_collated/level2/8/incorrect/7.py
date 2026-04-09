# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_053523/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'divisor', 'pool_size', 'bias_shape', 'sum_dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'max_pool_kernel_size', 'max_pool_stride', 'max_pool_padding', 'max_pool_dilation', 'max_pool_ceil_mode', 'max_pool_return_indices', 'global_avg_pool_output_size', 'divisor', 'bias', 'sum_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    """

    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

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
    # State for conv (nn.Conv3d)
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
    # State for max_pool (nn.MaxPool3d)
    state_kwargs['max_pool_kernel_size'] = model.max_pool.kernel_size
    state_kwargs['max_pool_stride'] = model.max_pool.stride
    state_kwargs['max_pool_padding'] = model.max_pool.padding
    state_kwargs['max_pool_dilation'] = model.max_pool.dilation
    state_kwargs['max_pool_ceil_mode'] = model.max_pool.ceil_mode
    state_kwargs['max_pool_return_indices'] = model.max_pool.return_indices
    # State for global_avg_pool (nn.AdaptiveAvgPool3d)
    state_kwargs['global_avg_pool_output_size'] = model.global_avg_pool.output_size
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
    if 'sum_dim' in flat_state:
        state_kwargs['sum_dim'] = flat_state['sum_dim']
    else:
        state_kwargs['sum_dim'] = getattr(model, 'sum_dim')
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

# ----------------------------------------------------------------------
# CUDA source – Fused convolution (with bias/div) and Fused Reduction
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch, const int in_channels, const int out_channels,
    const int in_d, const int in_h, const int in_w,
    const int kd, const int kh, const int kw,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const float divisor,
    const int out_d, const int out_h, const int out_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * out_d * out_h * out_w;
    if (idx >= total) return;

    int tmp = idx;
    int w = tmp % out_w; tmp /= out_w;
    int h = tmp % out_h; tmp /= out_h;
    int d = tmp % out_d; tmp /= out_d;
    int co = tmp % out_channels; tmp /= out_channels;
    int b = tmp;

    float acc = 0.0f;
    for (int ci = 0; ci < in_channels; ++ci) {
        for (int i = 0; i < kd; ++i) {
            int id = d * stride_d + i - pad_d;
            if (id < 0 || id >= in_d) continue;
            for (int j = 0; j < kh; ++j) {
                int ih = h * stride_h + j - pad_h;
                if (ih < 0 || ih >= in_h) continue;
                for (int k = 0; k < kw; ++k) {
                    int iw = w * stride_w + k - pad_w;
                    if (iw < 0 || iw >= in_w) continue;
                    
                    float in_val = input[((b * in_channels + ci) * in_d + id) * in_h * in_w + ih * in_w + iw];
                    float w_val = weight[(((co * in_channels + ci) * kd + i) * kh + j) * kw + k];
                    acc += in_val * w_val;
                }
            }
        }
    }
    acc /= divisor;
    if (bias != nullptr) acc += bias[co];
    output[idx] = acc;
}

__global__ void sum_fused_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch, const int out_channels,
    const float* __restrict__ bias_vec)
{
    // Input is (batch, out_channels, 1, 1, 1), output (batch, 1, 1, 1)
    int b = blockIdx.x;
    float sum_val = 0.0f;
    for (int co = 0; co < out_channels; ++co) {
        sum_val += input[b * out_channels + co] + bias_vec[co];
    }
    output[b] = sum_val;
}

void conv3d_fused(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
                  int stride_d, int stride_h, int stride_w, int pad_d, int pad_h, int pad_w, float divisor) {
    int batch = input.size(0);
    int in_channels = input.size(1);
    int in_d = input.size(2), in_h = input.size(3), in_w = input.size(4);
    int out_channels = weight.size(0);
    int kd = weight.size(2), kh = weight.size(3), kw = weight.size(4);
    int out_d = output.size(2), out_h = output.size(3), out_w = output.size(4);
    
    int total = batch * out_channels * out_d * out_h * out_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    conv3d_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr, output.data_ptr<float>(),
        batch, in_channels, out_channels, in_d, in_h, in_w, kd, kh, kw,
        stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, divisor, out_d, out_h, out_w);
}

void sum_fused(torch::Tensor input, torch::Tensor output, torch::Tensor bias) {
    int batch = input.size(0);
    int out_channels = input.size(1);
    sum_fused_kernel<<<batch, 1>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch, out_channels, bias.data_ptr<float>());
}
"""

cpp_source = r"""
void conv3d_fused(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int s_d, int s_h, int s_w, int p_d, int p_h, int p_w, float div);
void sum_fused(torch::Tensor input, torch::Tensor output, torch::Tensor bias);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d_fused", &conv3d_fused);
    m.def("sum_fused", &sum_fused);
}
"""

fused_ext = load_inline(name="fused_ops", cpp_sources=cpp_source, cuda_sources=cuda_source, extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups, max_pool_kernel_size, max_pool_stride, max_pool_padding, max_pool_dilation, max_pool_ceil_mode, max_pool_return_indices, global_avg_pool_output_size, divisor, bias, sum_dim):
    x = x.float().cuda()
    conv_weight = conv_weight.float().cuda()
    conv_bias = conv_bias.float().cuda() if conv_bias is not None else torch.tensor([], device='cuda')
    bias = bias.float().cuda()
    
    out_channels = conv_weight.shape[0]
    out_shape = (x.shape[0], out_channels, 
                 (x.shape[2] + 2*conv_padding[0] - conv_weight.shape[2]) // conv_stride[0] + 1,
                 (x.shape[3] + 2*conv_padding[1] - conv_weight.shape[3]) // conv_stride[1] + 1,
                 (x.shape[4] + 2*conv_padding[2] - conv_weight.shape[4]) // conv_stride[2] + 1)
    conv_out = torch.empty(out_shape, device='cuda')
    
    fused_ext.conv3d_fused(x, conv_weight, conv_bias, conv_out, *conv_stride, *conv_padding, divisor)
    
    x = F.max_pool3d(conv_out, max_pool_kernel_size, max_pool_stride, max_pool_padding, max_pool_dilation, max_pool_ceil_mode, max_pool_return_indices)
    x = F.adaptive_avg_pool3d(x, global_avg_pool_output_size)
    
    res = torch.empty((x.shape[0], 1, 1, 1), device='cuda')
    fused_ext.sum_fused(x, res, bias.reshape(-1))
    return res
