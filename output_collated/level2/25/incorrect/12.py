# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_082041/code_12.py
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

# Optimization: Fused CUDA kernel performing Implicit GEMM convolution,
# per-channel min reduction, and double tanh activation in a single pass.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" __global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
    float* __restrict__ out, int N, int Ci, int Co, int H, int W, int K, int pad) {

    // Tile configuration
    const int OH = H + 2 * pad - K + 1;
    const int OW = W + 2 * pad - K + 1;
    
    int n = blockIdx.z;
    int oh = blockIdx.y;
    int ow = blockIdx.x;

    if (oh >= OH || ow >= OW) return;

    // We store the min across channels in a register
    float min_val = 1e20f;
    
    // Iterate over Co
    for (int co = 0; co < Co; ++co) {
        float sum = b[co];
        for (int ci = 0; ci < Ci; ++ci) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int ih = oh + kh - pad;
                    int iw = ow + kw - pad;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        float val = x[((n * Ci + ci) * H + ih) * W + iw];
                        float weight = w[((co * Ci + ci) * K + kh) * K + kw];
                        sum += val * weight;
                    }
                }
            }
        }
        if (sum < min_val) min_val = sum;
    }
    
    // Apply double Tanh
    float t1 = tanhf(min_val);
    out[((n * 1 + 0) * OH + oh) * OW + ow] = tanhf(t1);
}

void fused_op_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out) {
    int N = x.size(0);
    int Ci = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int Co = w.size(0);
    int K = w.size(2);
    int pad = 1; // Fixed for 3x3 kernel
    int OH = H + 2 * pad - K + 1;
    int OW = W + 2 * pad - K + 1;

    dim3 blocks(OW, OH, N);
    fused_conv_min_tanh_kernel<<<blocks, 1>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
        out.data_ptr<float>(), N, Ci, Co, H, W, K, pad);
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv-Min-Tanh");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    # Determine output dimensions
    H, W = x.shape[2], x.shape[3]
    K = conv_weight.shape[2]
    OH = (H + 2 * conv_padding - K) // conv_stride + 1
    OW = (W + 2 * conv_padding - K) // conv_stride + 1
    
    # Initialize output tensor
    out = torch.empty((x.shape[0], 1, OH, OW), device=x.device, dtype=x.dtype)
    
    # Invoke custom CUDA kernel
    fused_ext.fused_op(x, conv_weight, conv_bias, out)
    
    return out
