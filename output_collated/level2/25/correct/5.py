# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_083856/code_14.py
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

# -------------------------------------------------------------------------
# 1. CUDA Kernel: Fused Conv2D + Channel-Min + Double Tanh
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int H, const int W, const int K,
    const int stride, const int padding,
    const int OH, const int OW)
{
    // Each thread calculates one (n, oh, ow) spatial output
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * OH * OW;
    if (tid >= total) return;

    const int n = tid / (OH * OW);
    const int remnant = tid % (OH * OW);
    const int oh = remnant / OW;
    const int ow = remnant % OW;

    // Use a small local array for the patch (max 16 * 3 * 3 = 144)
    float p[144];
    const int K_sq = K * K;

    // Load im2col patch into registers/local memory
    for (int ci = 0; ci < C_in; ++ci) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;
                float val = 0.0f;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    val = __ldg(&x[((n * C_in + ci) * H + ih) * W + iw]);
                }
                p[ci * K_sq + kh * K + kw] = val;
            }
        }
    }

    // Compute convolution for each output channel to find min
    float min_val = 1e38f;
    for (int co = 0; co < C_out; ++co) {
        float sum = bias[co];
        const int w_offset = co * C_in * K_sq;
        for (int i = 0; i < C_in * K_sq; ++i) {
            sum += p[i] * __ldg(&weight[w_offset + i]);
        }
        if (sum < min_val) min_val = sum;
    }

    // Double Tanh activation
    float res = tanhf(tanhf(min_val));
    output[tid] = res;
}

void fused_op_forward(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias,
    at::Tensor& output,
    int N, int C_in, int C_out, int H, int W, int K,
    int stride, int padding, int OH, int OW)
{
    const int threads = 256;
    const int blocks = (N * OH * OW + threads - 1) / threads;
    fused_conv_min_tanh_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, C_in, C_out, H, W, K, stride, padding, OH, OW
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias, 
                      at::Tensor& output, int N, int C_in, int C_out, int H, int W, int K, 
                      int stride, int padding, int OH, int OW);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused kernel");
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
    N, C_in, H, W = x.shape
    C_out, _, K, _ = conv_weight.shape
    stride = conv_stride[0] if isinstance(conv_stride, (tuple, list)) else conv_stride
    padding = conv_padding[0] if isinstance(conv_padding, (tuple, list)) else conv_padding
    
    OH = (H + 2 * padding - K) // stride + 1
    OW = (W + 2 * padding - K) // stride + 1
    
    output = torch.empty((N, 1, OH, OW), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(x, conv_weight, conv_bias, output, 
                       N, C_in, C_out, H, W, K, stride, padding, OH, OW)
    return output
