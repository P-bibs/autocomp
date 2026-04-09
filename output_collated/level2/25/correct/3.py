# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_083021/code_3.py
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

# CUDA Kernel: Fused Conv2D + Channel-Min + Tanh reduction
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_SIZE 16

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int B, int C, int H, int W,
    int OC, int K,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w
) {
    // Output spatial coordinates
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    
    if (out_h >= (H + 2 * pad_h - dilation_h * (K - 1) - 1) / stride_h + 1 ||
        out_w >= (W + 2 * pad_w - dilation_w * (K - 1) - 1) / stride_w + 1) return;

    float min_val = INFINITY;
    
    // Iterate over output channels to compute conv and find min
    for (int oc = 0; oc < OC; ++oc) {
        float sum = 0.0f;
        
        // Convolve with kernel
        for (int c = 0; c < C; ++c) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int ih = out_h * stride_h - pad_h + kh * dilation_h;
                    int iw = out_w * stride_w - pad_w + kw * dilation_w;
                    
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        float x_val = x[(((b * C + c) * H) + ih) * W + iw];
                        float w_val = w[(((oc * C + c) * K + kh) * K) + kw];
                        sum += x_val * w_val;
                    }
                }
            }
        }
        
        // Add bias
        sum += bias[oc];
        
        // Update minimum
        if (sum < min_val) min_val = sum;
    }
    
    // Apply double tanh
    float t = tanhf(min_val);
    out[(((b * 1) * ((H + 2 * pad_h - dilation_h * (K - 1) - 1) / stride_h + 1)) + out_h) * 
        ((W + 2 * pad_w - dilation_w * (K - 1) - 1) / stride_w + 1) + out_w] = tanhf(t);
}

void fused_op_forward(
    torch::Tensor x, torch::Tensor w, torch::Tensor bias, torch::Tensor out,
    int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w
) {
    int B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int OC = w.size(0), K = w.size(2);
    
    int out_height = (H + 2 * pad_h - dilation_h * (K - 1) - 1) / stride_h + 1;
    int out_width = (W + 2 * pad_w - dilation_w * (K - 1) - 1) / stride_w + 1;
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_width + block.x - 1) / block.x,
              (out_height + block.y - 1) / block.y,
              B);
    
    fused_conv_min_tanh_kernel<<<grid, block>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), B, C, H, W, OC, K,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor x, torch::Tensor w, torch::Tensor bias, torch::Tensor out,
    int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv2D + Min + Double Tanh");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
):
    # Handle parameter unpacking
    if isinstance(conv_stride, int):
        stride_h = stride_w = conv_stride
    else:
        stride_h, stride_w = conv_stride
    
    if isinstance(conv_padding, int):
        pad_h = pad_w = conv_padding
    else:
        pad_h, pad_w = conv_padding
        
    if isinstance(conv_dilation, int):
        dilation_h = dilation_w = conv_dilation
    else:
        dilation_h, dilation_w = conv_dilation
    
    B, C, H, W = x.shape
    OC, _, K, _ = conv_weight.shape
    
    # Calculate output dimensions
    out_height = (H + 2 * pad_h - dilation_h * (K - 1) - 1) // stride_h + 1
    out_width = (W + 2 * pad_w - dilation_w * (K - 1) - 1) // stride_w + 1
    
    out = torch.empty((B, 1, out_height, out_width), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(x, conv_weight, conv_bias, out, 
                       stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w)
    return out
