# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_085904/code_15.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies minimum operation, Tanh, and another Tanh.
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

# ----------------------------------------------------------------------
# CUDA source – fused kernel + host wrapper
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int H, const int W, const int K,
    const int stride, const int padding, const int dilation,
    const int H_out, const int W_out)
{
    const int total = N * H_out * W_out;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const int n = tid / (H_out * W_out);
    const int rem = tid % (H_out * W_out);
    const int oh = rem / W_out;
    const int ow = rem % W_out;

    // Convolve with every output channel
    float min_val = 1e38f;
    
    for (int oc = 0; oc < C_out; ++oc) {
        float sum = (bias != nullptr) ? bias[oc] : 0.0f;
        const int w_base = oc * C_in * K * K;
        
        for (int ic = 0; ic < C_in; ++ic) {
            const int in_ch_offset = (n * C_in + ic) * H * W;
            const int w_ch_offset = w_base + ic * K * K;
            
            for (int ky = 0; ky < K; ++ky) {
                int ih = oh * stride + ky * dilation - padding;
                if (ih < 0 || ih >= H) continue;
                
                for (int kx = 0; kx < K; ++kx) {
                    int iw = ow * stride + kx * dilation - padding;
                    if (iw >= 0 && iw < W) {
                        float in_v = __ldg(&input[in_ch_offset + ih * W + iw]);
                        float w_v = __ldg(&weight[w_ch_offset + ky * K + kx]);
                        sum += in_v * w_v;
                    }
                }
            }
        }
        if (sum < min_val) min_val = sum;
    }

    // Apply tanh twice: tanh(tanh(x))
    float a = tanhf(min_val);
    output[tid] = tanhf(a);
}

void fused_conv_min_tanh(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride, int padding, int dilation, int H_out, int W_out,
    torch::Tensor output)
{
    const int N = input.size(0), C_in = input.size(1), H = input.size(2), W = input.size(3);
    const int C_out = weight.size(0), K = weight.size(2);
    const int total = N * H_out * W_out;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    fused_conv_min_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.numel() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, C_out, H, W, K, stride, padding, dilation, H_out, W_out);
}
"""

cpp_source = r"""
void fused_conv_min_tanh(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                         int stride, int padding, int dilation, int H_out, int W_out,
                         torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_min_tanh", &fused_conv_min_tanh, "Fused conv-min-tanh");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    # Flatten stride/padding if provided as tuples/lists
    def to_int(v): return v[0] if hasattr(v, '__len__') else v
    stride, padding, dilation = to_int(conv_stride), to_int(conv_padding), to_int(conv_dilation)
    
    K = conv_weight.shape[2]
    H_out = (x.size(2) + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    W_out = (x.size(3) + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    
    out = torch.empty((x.size(0), 1, H_out, W_out), dtype=x.dtype, device=x.device)
    bias = conv_bias if conv_bias is not None else torch.empty(0, device=x.device)
    
    fused_ext.fused_conv_min_tanh(x, conv_weight, bias, stride, padding, dilation, H_out, W_out, out)
    return out

def get_init_inputs(): return [16, 64, 3]
def get_inputs(): return [torch.rand(128, 16, 256, 256, device='cuda')]
