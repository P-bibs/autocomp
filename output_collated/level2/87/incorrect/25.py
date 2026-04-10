# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_142828/code_11.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'subtract_value_1', 'subtract_value_2']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'subtract_value_1', 'subtract_value_2']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, subtracts two values, applies Mish activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

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
    if 'subtract_value_1' in flat_state:
        state_kwargs['subtract_value_1'] = flat_state['subtract_value_1']
    else:
        state_kwargs['subtract_value_1'] = getattr(model, 'subtract_value_1')
    if 'subtract_value_2' in flat_state:
        state_kwargs['subtract_value_2'] = flat_state['subtract_value_2']
    else:
        state_kwargs['subtract_value_2'] = getattr(model, 'subtract_value_2')
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

# --- CUDA Kernel Code ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float softplus(float x) {
    // Use numerically stable version of softplus
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__global__ void fused_conv_mish_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float sub1,
    float sub2,
    float* __restrict__ y,
    int batch,
    int in_ch,
    int out_ch,
    int H,
    int W,
    int out_H,
    int out_W
) {
    // Thread indices
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;

    if (ox >= out_W || oy >= out_H) return;

    const int kH = 3;
    const int kW = 3;
    const int pad_h = 1;
    const int pad_w = 1;
    const int stride_h = 1;
    const int stride_w = 1;
    const int dilation_h = 1;
    const int dilation_w = 1;

    // Output index
    int out_idx = ((/* batch index */ 0) * out_ch + oc) * out_H * out_W + oy * out_W + ox;

    // Convolution accumulator
    float acc = 0.0f;

    // Loop over input channels and kernel positions
    for (int ic = 0; ic < in_ch; ++ic) {
        for (int ky = 0; ky < kH; ++ky) {
            for (int kx = 0; kx < kW; ++kx) {
                // Calculate input coordinates
                int iy = oy * stride_h - pad_h + ky * dilation_h;
                int ix = ox * stride_w - pad_w + kx * dilation_w;

                // Check bounds
                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                    int x_idx = (/* batch index */ 0) * in_ch * H * W + ic * H * W + iy * W + ix;
                    int w_idx = oc * in_ch * kH * kW + ic * kH * kW + ky * kW + kx;
                    acc += x[x_idx] * weight[w_idx];
                }
            }
        }
    }

    // Add bias
    acc += bias[oc];

    // Subtract constants
    acc -= sub1;
    acc -= sub2;

    // Apply Mish activation: y = x * tanh(softplus(x))
    float sp = softplus(acc);
    y[out_idx] = acc * tanhf(sp);
}

void fused_conv_mish_forward(
    int batch,
    int in_ch,
    int out_ch,
    int H,
    int W,
    int out_H,
    int out_W,
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float sub1,
    float sub2,
    torch::Tensor& y
) {
    // Get raw pointers
    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    const float* b_ptr = bias.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();

    // Launch configuration
    const int TILE_W = 16;
    const int TILE_H = 16;
    dim3 block(TILE_W, TILE_H);
    dim3 grid((out_W + TILE_W - 1) / TILE_W, (out_H + TILE_H - 1) / TILE_H, batch * out_ch);

    // Launch kernel for each sample in the batch
    for (int n = 0; n < batch; ++n) {
        fused_conv_mish_forward_kernel<<<grid, block>>>(
            x_ptr + n * in_ch * H * W,
            w_ptr,
            b_ptr,
            sub1,
            sub2,
            y_ptr + n * out_ch * out_H * out_W,
            1, // batch size per kernel call
            in_ch,
            out_ch,
            H,
            W,
            out_H,
            out_W
        );
    }
    cudaDeviceSynchronize(); // Ensure completion before returning
}
"""

# --- C++ Binding Code ---
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_mish_forward(
    int batch,
    int in_ch,
    int out_ch,
    int H,
    int W,
    int out_H,
    int out_W,
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float sub1,
    float sub2,
    torch::Tensor& y
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_mish", &fused_conv_mish_forward, "Fused Convolution + Mish activation");
}
"""

# --- Compile Extension ---
fused_ext = load_inline(
    name='fused_conv_mish_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# --- Optimized Python Function ---
def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
    subtract_value_1,
    subtract_value_2,
):
    # Sanity checks – the current model uses stride=1, padding=1, dilation=1, groups=1
    assert conv_stride == (1, 1)
    assert conv_padding == (1, 1)
    assert conv_dilation == (1, 1)
    assert conv_groups == 1

    batch, _, H, W = x.shape
    out_ch, in_ch, kH, kW = conv_weight.shape
    out_H = (H + 2 * conv_padding[0] - conv_dilation[0] * (kH - 1) - 1) // conv_stride[0] + 1
    out_W = (W + 2 * conv_padding[1] - conv_dilation[1] * (kW - 1) - 1) // conv_stride[1] + 1

    # Allocate output tensor (contiguous, float32)
    y = torch.empty((batch, out_ch, out_H, out_W), dtype=x.dtype, device=x.device)

    # Call the fused kernel
    fused_ext.fused_conv_mish(
        batch,
        in_ch,
        out_ch,
        H,
        W,
        out_H,
        out_W,
        x,
        conv_weight,
        conv_bias,
        float(subtract_value_1),
        float(subtract_value_2),
        y,
    )
    return y


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
