# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_153448/code_6.py
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
# CUDA kernel: fused transposed-convolution + add -> clamp -> GELU -> mul
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Fast GELU approximation used by many CUDA libraries
__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Fused kernel: transposed convolution followed by the element-wise chain
__global__ void fused_conv_transpose_kernel(
    const float* __restrict__ input,      // (N, Cin, Hin, Win)
    const float* __restrict__ weight,     // (Cout, Cin, K, K)
    const float* __restrict__ bias,       // (Cout) or nullptr
    float* __restrict__ output,           // (N, Cout, Hout, Wout)
    const int N, const int Cin, const int Hin, const int Win,
    const int Cout, const int K,
    const int stride, const int padding, const int output_padding,
    const int dilation,
    const float add_val, const float mul_val,
    const int Hout, const int Wout)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * Cout * Hout * Wout;
    if (idx >= total) return;

    // ---- decode flat index to (n, oc, oh, ow) ----
    int remaining = idx;
    const int ow = remaining % Wout;
    remaining /= Wout;
    const int oh = remaining % Hout;
    remaining /= Hout;
    const int oc = remaining % Cout;
    const int n  = remaining / Cout;

    // ---- transposed convolution accumulation ----
    float sum = 0.0f;

    // loop over input channels and kernel positions
    for (int ci = 0; ci < Cin; ++ci) {
        // base index for the weight of output channel oc and input channel ci
        const int wbase = ((oc * Cin + ci) * K) * K;

        for (int kh = 0; kh < K; ++kh) {
            // compute the input row coordinate
            int ih = (oh + padding - kh * dilation);
            if (ih % stride != 0) continue;
            ih /= stride;
            if (ih < 0 || ih >= Hin) continue;

            for (int kw = 0; kw < K; ++kw) {
                // compute the input column coordinate
                int iw = (ow + padding - kw * dilation);
                if (iw % stride != 0) continue;
                iw /= stride;
                if (iw < 0 || iw >= Win) continue;

                // input value
                const float inp_val = input[((n * Cin + ci) * Hin + ih) * Win + iw];
                // weight value
                const float w_val = weight[wbase + kh * K + kw];
                sum += inp_val * w_val;
            }
        }
    }

    // optional bias
    if (bias != nullptr) sum += bias[oc];

    // ---- fused element-wise chain ----
    float val = sum + add_val;
    val = fminf(val, 0.0f);            // clamp to <=0
    val = fast_gelu(val);
    val = val * mul_val;

    // write final result
    output[((n * Cout + oc) * Hout + oh) * Wout + ow] = val;
}

// C++ wrapper that launches the kernel
void fused_conv_transpose(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int N, int Cin, int Hin, int Win,
    int Cout, int K,
    int stride, int padding, int output_padding,
    int dilation,
    float add_val,
    float mul_val,
    int Hout, int Wout)
{
    const int total = N * Cout * Hout * Wout;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    fused_conv_transpose_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, Cin, Hin, Win,
        Cout, K,
        stride, padding, output_padding,
        dilation,
        add_val, mul_val,
        Hout, Wout);
}
"""

# ----------------------------------------------------------------------
# C++ binding (PYBIND11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int N, int Cin, int Hin, int Win,
    int Cout, int K,
    int stride, int padding, int output_padding,
    int dilation,
    float add_val,
    float mul_val,
    int Hout, int Wout);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose", &fused_conv_transpose,
          "Fused transposed-convolution + add-clamp-gelu-mul");
}
"""

# ----------------------------------------------------------------------
# Compile the extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_conv_transpose',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Model parameters (taken from the original script)
# ----------------------------------------------------------------------
batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

# Default padding / output-padding / dilation used in the original call
conv_transpose_padding = 0          # default for F.conv_transpose2d
conv_transpose_output_padding = 0  # default
conv_transpose_dilation = 1        # default
conv_transpose_groups = 1          # not used in this implementation

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride,
            add_value, multiply_value]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]

# ----------------------------------------------------------------------
# Functional model – replaces the original conv_transpose + fused kernel
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
    # Compute output spatial size using the transposed-conv formula
    # ------------------------------------------------------------------
    N = x.size(0)
    Cin = x.size(1)
    Hin = x.size(2)
    Win = x.size(3)

    Cout = conv_transpose_weight.size(0)      # out_channels
    K = conv_transpose_weight.size(2)        # kernel size (square)

    Hout = (Hin - 1) * conv_transpose_stride \
           - 2 * conv_transpose_padding \
           + conv_transpose_dilation * (K - 1) \
           + conv_transpose_output_padding + 1

    Wout = (Win - 1) * conv_transpose_stride \
           - 2 * conv_transpose_padding \
           + conv_transpose_dilation * (K - 1) \
           + conv_transpose_output_padding + 1

    # Allocate output tensor
    output = torch.empty((N, Cout, Hout, Wout), dtype=torch.float32, device='cuda')

    # ------------------------------------------------------------------
    # Launch the fused kernel
    # ------------------------------------------------------------------
    fused_ext.fused_conv_transpose(
        x,                       # input
        conv_transpose_weight,   # weight
        conv_transpose_bias if conv_transpose_bias is not None else torch.empty(0),
        output,
        N, Cin, Hin, Win,
        Cout, K,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_dilation,
        float(add_value),
        float(multiply_value),
        Hout, Wout
    )
    return output
