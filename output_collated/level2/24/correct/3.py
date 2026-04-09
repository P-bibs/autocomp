# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_100332/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a 3D convolution, applies minimum operation along a specific dimension, 
    and then applies softmax.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

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
    if 'dim' in flat_state:
        state_kwargs['dim'] = flat_state['dim']
    else:
        state_kwargs['dim'] = getattr(model, 'dim')
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

# ----------------------------------------------------------------------
# CUDA Kernel: Fused 3D Conv + Min Reduction
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_min_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int in_d, const int in_h, const int in_w,
    const int kd, const int kh, const int kw,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int groups,
    const int out_d, const int out_h, const int out_w)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C_out * out_h * out_w;
    if (idx >= total) return;

    int n = idx / (C_out * out_h * out_w);
    int rem = idx % (C_out * out_h * out_w);
    int co = rem / (out_h * out_w);
    int rem2 = rem % (out_h * out_w);
    int h = rem2 / out_w;
    int w = rem2 % out_w;

    int group_size = C_in / groups;
    int group_start_ic = (co / (C_out / groups)) * group_size;

    float min_val = 3.40282e+38f; // FLT_MAX

    for (int od = 0; od < out_d; ++od) {
        float acc = (bias != nullptr) ? bias[co] : 0.0f;
        for (int ic = 0; ic < group_size; ++ic) {
            int input_c = group_start_ic + ic;
            for (int kid = 0; kid < kd; ++kid) {
                int id = od * stride_d + kid - pad_d;
                if (id < 0 || id >= in_d) continue;
                for (int kih = 0; kih < kh; ++kih) {
                    int ih = h * stride_h + kih - pad_h;
                    if (ih < 0 || ih >= in_h) continue;
                    for (int kiw = 0; kiw < kw; ++kiw) {
                        int iw = w * stride_w + kiw - pad_w;
                        if (iw < 0 || iw >= in_w) continue;
                        
                        float w_val = weight[((co * group_size + ic) * kd + kid) * kh * kw + kih * kw + kiw];
                        float i_val = input[((n * C_in + input_c) * in_d + id) * in_h * in_w + ih * in_w + iw];
                        acc += i_val * w_val;
                    }
                }
            }
        }
        if (acc < min_val) min_val = acc;
    }
    output[idx] = min_val;
}

void launch_fused_conv3d_min(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w, int groups) 
{
    const int N = input.size(0), C_in = input.size(1), C_out = weight.size(0);
    const int in_d = input.size(2), in_h = input.size(3), in_w = input.size(4);
    const int kd = weight.size(2), kh = weight.size(3), kw = weight.size(4);
    const int out_d = output.size(2), out_h = output.size(2), out_w = output.size(3);
    
    // Note: In original task, we reduce over depth dim=2, so output spatial is (H', W')
    int total = N * C_out * output.size(2) * output.size(3);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    const float* b_ptr = (bias.numel() > 0) ? bias.data_ptr<float>() : nullptr;
    
    conv3d_min_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), b_ptr, output.data_ptr<float>(),
        N, C_in, C_out, in_d, in_h, in_w, kd, kh, kw,
        stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, groups,
        (in_d + 2*pad_d - kd) / stride_d + 1, output.size(2), output.size(3)
    );
}
"""

cpp_source = """
void launch_fused_conv3d_min(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, int, int, int, int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fused_op", &launch_fused_conv3d_min); }
"""

fused_ext = load_inline(name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source, 
                        extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, 
                     conv_dilation, conv_groups, dim=2):
    # Output spatial dims after reduction of dim 2
    stride = conv_stride if isinstance(conv_stride, tuple) else (conv_stride,)*3
    padding = conv_padding if isinstance(conv_padding, tuple) else (conv_padding,)*3
    out_h = (x.shape[3] + 2 * padding[1] - conv_weight.shape[3]) // stride[1] + 1
    out_w = (x.shape[4] + 2 * padding[2] - conv_weight.shape[4]) // stride[2] + 1
    
    output = torch.empty((x.shape[0], conv_weight.shape[0], out_h, out_w), device=x.device)
    fused_ext.fused_op(x, conv_weight, conv_bias if conv_bias is not None else torch.tensor([]), 
                       output, *stride, *padding, conv_groups)
    return torch.softmax(output, dim=1)
