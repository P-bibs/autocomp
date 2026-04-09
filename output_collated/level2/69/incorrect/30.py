# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_051902/code_7.py
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

# -------------------------------------------------------------------------
#  CUDA source – fused convolution + hardswish + ReLU kernel
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_fused_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int H_in, const int W_in,
    const int H_out, const int W_out,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int padding_h, const int padding_w,
    const int dilation_h, const int dilation_w,
    const int groups,
    const int C_in_per_group, const int C_out_per_group)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    // Decode flat index to (n, oc, oh, ow)
    int oc = idx % C_out;
    int tmp = idx / C_out;
    int ow = tmp % W_out;
    tmp /= W_out;
    int oh = tmp % H_out;
    int n = tmp / H_out;

    int group_id = oc / C_out_per_group;
    int ic_start = group_id * C_in_per_group;

    float sum = 0.0f;
    if (bias != nullptr) {
        sum = __half2float(bias[oc]);
    }

    // Convolution: compute sum for this output location
    for (int ic = 0; ic < C_in_per_group; ++ic) {
        int ic_idx = ic_start + ic;
        for (int kh = 0; kh < kernel_h; ++kh) {
            int ih = oh * stride_h - padding_h + kh * dilation_h;
            if (ih < 0 || ih >= H_in) continue;
            for (int kw = 0; kw < kernel_w; ++kw) {
                int iw = ow * stride_w - padding_w + kw * dilation_w;
                if (iw < 0 || iw >= W_in) continue;
                
                int in_idx = ((n * C_in + ic_idx) * H_in + ih) * W_in + iw;
                // Using __ldg for read-only cache optimization
                half in_val = __ldg(&input[in_idx]);
                int wt_idx = ((oc * C_in_per_group + ic) * kernel_h + kh) * kernel_w + kw;
                half w_val = __ldg(&weight[wt_idx]);
                
                sum += __half2float(in_val) * __half2float(w_val);
            }
        }
    }

    // Fused activation: hardswish (x * clamp(x+3, 0, 6) / 6) then relu
    float hs = sum;
    float tmp_hs = hs + 3.0f;
    float clamped = (tmp_hs > 6.0f) ? 6.0f : ((tmp_hs < 0.0f) ? 0.0f : tmp_hs);
    hs = (hs * clamped) * 0.16666667f; // Multiply by 1/6
    
    // ReLU
    output[idx] = __float2half(hs > 0.0f ? hs : 0.0f);
}

void fused_op_forward(int blocks, int threads, 
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int N, int C_in, int C_out, int H_in, int W_in, int H_out, int W_out,
    int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw, int groups, int CinG, int CoutG) 
{
    conv_fused_kernel<<<blocks, threads>>>(
        (const half*)input.data_ptr(), (const half*)weight.data_ptr(), 
        bias.numel() > 0 ? (const half*)bias.data_ptr() : nullptr, 
        (half*)output.data_ptr(), N, C_in, C_out, H_in, W_in, H_out, W_out,
        kh, kw, sh, sw, ph, pw, dh, dw, groups, CinG, CoutG
    );
}
"""

cpp_source = r"""
void fused_op_forward(int blocks, int threads, torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int N, int C_in, int C_out, int H_in, int W_in, int H_out, int W_out,
    int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw, int groups, int CinG, int CoutG);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused convolution + hardswish + ReLU");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    device = x.device
    stride = (conv_stride, conv_stride) if isinstance(conv_stride, int) else conv_stride
    pad = (conv_padding, conv_padding) if isinstance(conv_padding, int) else conv_padding
    dil = (conv_dilation, conv_dilation) if isinstance(conv_dilation, int) else conv_dilation
    
    x = x.to(device).half().contiguous()
    weight = conv_weight.to(device).half().contiguous()
    bias = conv_bias.to(device).half().contiguous() if conv_bias is not None else torch.empty(0, device=device, dtype=torch.half)
    
    N, C_in, H_in, W_in = x.shape
    C_out = weight.shape[0]
    kh, kw = weight.shape[2], weight.shape[3]
    H_out = (H_in + 2 * pad[0] - dil[0] * (kh - 1) - 1) // stride[0] + 1
    W_out = (W_in + 2 * pad[1] - dil[1] * (kw - 1) - 1) // stride[1] + 1
    
    output = torch.empty((N, C_out, H_out, W_out), dtype=torch.half, device=device)
    
    threads = 256
    blocks = (N * C_out * H_out * W_out + threads - 1) // threads
    
    fused_ext.fused_op(blocks, threads, x, weight, bias, output,
                       N, C_in, C_out, H_in, W_in, H_out, W_out,
                       kh, kw, stride[0], stride[1], pad[0], pad[1], dil[0], dil[1],
                       conv_groups, C_in // conv_groups, C_out // conv_groups)
    
    return output.float()
