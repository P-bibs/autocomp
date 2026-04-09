# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_051208/code_3.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- CUDA Kernel (Fused Conv + HardSwish + ReLU) ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_activation_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size,
    int stride, int padding, int dilation, int groups,
    int out_height, int out_width
) {
    // Grid: (batch, out_channels, out_height, out_width)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_points = batch_size * out_channels * out_height * out_width;
    if (idx >= total_points) return;

    // Unpack indices
    int n = idx / (out_channels * out_height * out_width);
    int rem = idx % (out_channels * out_height * out_width);
    int oc = rem / (out_height * out_width);
    rem = rem % (out_height * out_width);
    int oh = rem / out_width;
    int ow = rem % out_width;

    // Convolution computation
    float sum = 0.0f;
    if (bias != nullptr) sum = bias[oc];

    int ic_start = (oc * in_channels) / groups;
    int ic_end = ic_start + in_channels / groups;

    for (int ic = ic_start; ic < ic_end; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            int ih = oh * stride + kh * dilation - padding;
            if (ih < 0 || ih >= height) continue;
            for (int kw = 0; kw < kernel_size; ++kw) {
                int iw = ow * stride + kw * dilation - padding;
                if (iw < 0 || iw >= width) continue;
                // Use __ldg for read-only cache
                float in_val = __ldg(&input[((n * in_channels + ic) * height + ih) * width + iw]);
                float w_val = __ldg(&weight[((oc * (in_channels / groups) + (ic - ic_start)) * kernel_size + kh) * kernel_size + kw]);
                sum += in_val * w_val;
            }
        }
    }

    // HardSwish: x * clamp(x + 3, 0, 6) / 6
    float x = sum;
    float hs = x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
    // ReLU: max(0, x)
    float out_val = fmaxf(hs, 0.0f);

    // Store result
    output[((n * out_channels + oc) * out_height + oh) * out_width + ow] = out_val;
}

void fused_conv_activation(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride, int padding, int dilation, int groups
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    int total_points = batch_size * out_channels * out_height * out_width;
    int block_size = 256;
    int grid_size = (total_points + block_size - 1) / block_size;

    fused_conv_activation_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        height, width, kernel_size,
        stride, padding, dilation, groups,
        out_height, out_width
    );
}
"""

# --- C++ Interface (PYBIND11) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_activation(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride, int padding, int dilation, int groups);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_activation", &fused_conv_activation,
          "Fused Conv2d + HardSwish + ReLU forward");
}
"""

# Compile inline CUDA extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# --- Optimized Functional Model ---
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
    # Compute output spatial dimensions
    kh = conv_weight.size(2)
    out_h = (x.size(2) + 2 * conv_padding - conv_dilation * (kh - 1) - 1) // conv_stride + 1
    out_w = (x.size(3) + 2 * conv_padding - conv_dilation * (kh - 1) - 1) // conv_stride + 1

    # Allocate output tensor on the same device as input
    out = torch.empty(x.size(0), conv_weight.size(0), out_h, out_w, dtype=x.dtype, device=x.device)

    # Call fused CUDA kernel
    fused_ext.fused_conv_activation(
        x, conv_weight, conv_bias, out,
        conv_stride, conv_padding, conv_dilation, conv_groups
    )
    return out

# --- Input Generation ---
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    # Generate input on GPU
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
