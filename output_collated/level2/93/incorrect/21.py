# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_153448/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'add_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'add_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

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
    # State for conv_transpose (nn.ConvTranspose2d)
    if 'conv_transpose_weight' in flat_state:
        state_kwargs['conv_transpose_weight'] = flat_state['conv_transpose_weight']
    else:
        state_kwargs['conv_transpose_weight'] = getattr(model.conv_transpose, 'weight', None)
    if 'conv_transpose_bias' in flat_state:
        state_kwargs['conv_transpose_bias'] = flat_state['conv_transpose_bias']
    else:
        state_kwargs['conv_transpose_bias'] = getattr(model.conv_transpose, 'bias', None)
    state_kwargs['conv_transpose_stride'] = model.conv_transpose.stride
    state_kwargs['conv_transpose_padding'] = model.conv_transpose.padding
    state_kwargs['conv_transpose_output_padding'] = model.conv_transpose.output_padding
    state_kwargs['conv_transpose_groups'] = model.conv_transpose.groups
    state_kwargs['conv_transpose_dilation'] = model.conv_transpose.dilation
    if 'add_value' in flat_state:
        state_kwargs['add_value'] = flat_state['add_value']
    else:
        state_kwargs['add_value'] = getattr(model, 'add_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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
# Inline CUDA source – the fused transposed-convolution kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// ------------------------------------------------------------------------
// Fused kernel: transposed convolution + add + min(.,0) + gelu + mul
// ------------------------------------------------------------------------
__global__ void deconv_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int H, const int W,
    const int C_out, const int K,
    const int stride, const int padding, const int /*output_padding*/, const int dilation,
    const int out_h, const int out_w,
    const float add_value,
    const float multiply_value,
    const int has_bias)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C_out * out_h * out_w;
    if (idx >= total) return;

    // decode linear index to (n, oc, oh, ow)
    int tmp = idx;
    int n = tmp / (C_out * out_h * out_w);
    tmp %= (C_out * out_h * out_w);
    int oc = tmp / (out_h * out_w);
    tmp %= (out_h * out_w);
    int oh = tmp / out_w;
    int ow = tmp % out_w;

    float sum = 0.0f;
    if (has_bias) sum += bias[oc];

    // ----- transposed convolution (naïve but GPU-parallel) -----
    for (int ic = 0; ic < C_in; ++ic) {
        const float* w_ic_oc = weight + (ic * C_out * K * K + oc * K * K);
        for (int ky = 0; ky < K; ++ky) {
            int i_h = oh * stride - padding + ky * dilation;
            if (i_h < 0 || i_h >= H) continue;
            for (int kx = 0; kx < K; ++kx) {
                int i_w = ow * stride - padding + kx * dilation;
                if (i_w < 0 || i_w >= W) continue;
                float inp_val = input[(n * C_in + ic) * H * W + i_h * W + i_w];
                float w_val = w_ic_oc[ky * K + kx];
                sum += inp_val * w_val;
            }
        }
    }

    // ----- fused point-wise ops -----
    float val = sum + add_value;
    // min(x, 0.0f)  →  ReLU that passes negatives
    if (val > 0.0f) val = 0.0f;

    // GELU approximation (same as PyTorch)
    float x = val;
    float tanh_arg = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    float tanh_val = tanhf(tanh_arg);
    val = 0.5f * x * (1.0f + tanh_val);

    val *= multiply_value;

    // write result
    output[(n * C_out + oc) * out_h * out_w + oh * out_w + ow] = val;
}

// ------------------------------------------------------------------------
// Host wrapper that launches the kernel
// ------------------------------------------------------------------------
void deconv_fused(
    int64_t input_ptr,
    int64_t weight_ptr,
    int64_t bias_ptr,
    int64_t output_ptr,
    int N, int C_in, int H, int W,
    int C_out, int K,
    int stride, int padding, int output_padding, int dilation,
    int out_h, int out_w,
    float add_value,
    float multiply_value,
    int has_bias)
{
    const float* input = reinterpret_cast<const float*>(input_ptr);
    const float* weight = reinterpret_cast<const float*>(weight_ptr);
    const float* bias = has_bias ? reinterpret_cast<const float*>(bias_ptr) : nullptr;
    float* output = reinterpret_cast<float*>(output_ptr);

    const int block_size = 256;
    const int total = N * C_out * out_h * out_w;
    const int grid = (total + block_size - 1) / block_size;

    deconv_fused_kernel<<<grid, block_size>>>(
        input, weight, bias, output,
        N, C_in, H, W,
        C_out, K,
        stride, padding, output_padding, dilation,
        out_h, out_w,
        add_value, multiply_value, has_bias);

    cudaDeviceSynchronize();
}
"""

# ----------------------------------------------------------------------
# Inline C++ (PyBind11) source – Python binding for the CUDA function
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void deconv_fused(
    int64_t input_ptr,
    int64_t weight_ptr,
    int64_t bias_ptr,
    int64_t output_ptr,
    int N, int C_in, int H, int W,
    int C_out, int K,
    int stride, int padding, int output_padding, int dilation,
    int out_h, int out_w,
    float add_value,
    float multiply_value,
    int has_bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &deconv_fused, "Fused transposed-convolution forward");
}
"""

# ----------------------------------------------------------------------
# Compile the extension (runs at import time)
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# The optimized functional_model – replaces the original CPU code
# ----------------------------------------------------------------------
def functional_model(
    x,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    add_value,
    multiply_value,
):
    # ------------------------------------------------------------------
    # Ensure all tensors are on the GPU
    # ------------------------------------------------------------------
    if not x.is_cuda:
        x = x.cuda()
    if not conv_transpose_weight.is_cuda:
        conv_transpose_weight = conv_transpose_weight.cuda()
    if conv_transpose_bias is not None and not conv_transpose_bias.is_cuda:
        conv_transpose_bias = conv_transpose_bias.cuda()

    # ------------------------------------------------------------------
    # Basic shape information
    # ------------------------------------------------------------------
    N, C_in, H, W = x.shape
    C_out = conv_transpose_weight.shape[1]
    K = conv_transpose_weight.shape[2]          # square kernel assumed
    stride = conv_transpose_stride
    padding = conv_transpose_padding
    output_padding = conv_transpose_output_padding
    dilation = conv_transpose_dilation

    # ------------------------------------------------------------------
    # Compute output spatial size (standard transposed-conv formula)
    # ------------------------------------------------------------------
    out_h = (H - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
    out_w = (W - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1

    # ------------------------------------------------------------------
    # Allocate output tensor on the GPU
    # ------------------------------------------------------------------
    output = torch.empty((N, C_out, out_h, out_w), dtype=torch.float32, device='cuda')

    # ------------------------------------------------------------------
    # Prepare bias pointer and flag
    # ------------------------------------------------------------------
    has_bias = 1 if conv_transpose_bias is not None else 0
    bias_ptr = conv_transpose_bias.data_ptr() if conv_transpose_bias is not None else 0

    # ------------------------------------------------------------------
    # Launch the fused CUDA kernel
    # ------------------------------------------------------------------
    fused_ext.fused_op(
        x.data_ptr(),
        conv_transpose_weight.data_ptr(),
        bias_ptr,
        output.data_ptr(),
        N, C_in, H, W,
        C_out, K,
        stride, padding, output_padding, dilation,
        out_h, out_w,
        add_value,
        multiply_value,
        has_bias)

    return output

# ----------------------------------------------------------------------
# Helper functions used by the test harness (optional)
# ----------------------------------------------------------------------
def get_init_inputs():
    # Return dummy init args – they are not needed for the functional model
    return [64, 128, 4, 2, 0.5, 2.0]

def get_inputs():
    batch_size = 128
    in_channels = 64
    height = width = 64
    return [torch.rand(batch_size, in_channels, height, width)]
