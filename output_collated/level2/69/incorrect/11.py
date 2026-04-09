# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_050905/code_3.py
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

# ----------------------------------------------------------------------
# 1.  CUDA kernel (convolution + hardswish + ReLU)  --------------------
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_relu_hardswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_size,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int out_height,
    const int out_width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_out = batch_size * out_channels * out_height * out_width;
    if (idx >= total_out) return;

    // decode linear index -> (n, oc, oh, ow)
    int n = idx / (out_channels * out_height * out_width);
    int rem = idx % (out_channels * out_height * out_width);
    int oc = rem / (out_height * out_width);
    int rem2 = rem % (out_height * out_width);
    int oh = rem2 / out_width;
    int ow = rem2 % out_width;

    // ---------- convolution ----------
    float sum = 0.0f;
    // groups == 1 is assumed for this example (the original call never uses groups > 1)
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = oh * stride_h - pad_h + kh * dilation_h;
                int iw = ow * stride_w - pad_w + kw * dilation_w;
                // zero‑padding boundary check
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int inp_idx = ((n * in_channels + ic) * height + ih) * width + iw;
                    int w_idx   = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                    sum += input[inp_idx] * weight[w_idx];
                }
            }
        }
    }

    // bias term
    if (bias != nullptr) sum += bias[oc];

    // ---------- activation: hardswish ----------
    float x = sum;
    float y = x + 3.0f;
    y = fminf(fmaxf(y, 0.0f), 6.0f);
    x = x * y * (1.0f / 6.0f);

    // ---------- activation: ReLU ----------
    x = fmaxf(0.0f, x);

    output[idx] = x;
}
"""

# ----------------------------------------------------------------------
# 2.  C++ wrapper that launches the kernel (PyBind11)  -----------------
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op,
          "Fused conv2d + hardswish + ReLU CUDA kernel");
}
"""

# Append the wrapper implementation to the CUDA source because the kernel
# definition must be visible to the compiler.
full_cuda_source = cuda_source + r"""
void fused_op(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups)
{
    // sizes
    int batch_size   = input.size(0);
    int in_channels  = input.size(1);
    int height       = input.size(2);
    int width        = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size  = weight.size(2);

    // output spatial size (standard convolution formula)
    int out_height = (height + 2*pad_h - dilation_h*(kernel_size - 1) - 1) / stride_h + 1;
    int out_width  = (width  + 2*pad_w - dilation_w*(kernel_size - 1) - 1) / stride_w + 1;

    // allocate output
    auto output = torch::empty({batch_size, out_channels, out_height, out_width},
                               input.options());

    const float* input_ptr  = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr   = bias.defined() ? bias.data_ptr<float>() : nullptr;
    float*       output_ptr = output.data_ptr<float>();

    // grid / block size
    const int block_dim = 256;
    int total_out = batch_size * out_channels * out_height * out_width;
    int grid_dim = (total_out + block_dim - 1) / block_dim;

    fused_conv_relu_hardswish_kernel<<<grid_dim, block_dim>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, stride_h, stride_w, pad_h, pad_w,
        dilation_h, dilation_w, groups, out_height, out_width);

    cudaDeviceSynchronize();   // needed because we return a tensor to Python
}
"""

# ----------------------------------------------------------------------
# 3.  Build the inline extension ---------------------------------------
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=full_cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# ----------------------------------------------------------------------
# 4.  Dummy sizes that match the original benchmark -------------------
# ----------------------------------------------------------------------
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width,
                       device='cuda', dtype=torch.float32)]

# ----------------------------------------------------------------------
# 5.  Optimised functional_model ---------------------------------------
# ----------------------------------------------------------------------
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
    """
    Fused CUDA implementation of:
        conv2d (stride, pad, dilation, groups) -> hardswish -> relu
    All three steps are executed in a single kernel to eliminate
    separate kernel launches and to keep the activation data in registers.
    """
    # ensure contiguous layout on the device
    x = x.contiguous()
    conv_weight = conv_weight.contiguous()
    if conv_bias is not None:
        conv_bias = conv_bias.contiguous()

    # normalise stride / padding / dilation to integer values
    if isinstance(conv_stride, (tuple, list)):
        stride_h = conv_stride[0]
        stride_w = conv_stride[1] if len(conv_stride) > 1 else conv_stride[0]
    else:
        stride_h = stride_w = conv_stride

    if isinstance(conv_padding, (tuple, list)):
        pad_h = conv_padding[0]
        pad_w = conv_padding[1] if len(conv_padding) > 1 else conv_padding[0]
    else:
        pad_h = pad_w = conv_padding

    if isinstance(conv_dilation, (tuple, list)):
        dilation_h = conv_dilation[0]
        dilation_w = conv_dilation[1] if len(conv_dilation) > 1 else conv_dilation[0]
    else:
        dilation_h = dilation_w = conv_dilation

    groups = conv_groups

    # invoke the fused kernel
    out = fused_ext.fused_op(
        x,
        conv_weight,
        conv_bias,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
    )
    return out
