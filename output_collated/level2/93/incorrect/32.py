# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_154319/code_7.py
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
# Inline CUDA source – fused transposed‑convolution + point‑wise ops
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fast gelu approximation: 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
__device__ __forceinline__ float fast_gelu(float x) {
    const float sqrt_2_over_pi = sqrtf(2.0f / 3.141592653589793f);
    float t = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(t));
}

__global__ void deconv_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int H_in, const int W_in,
    const int H_out, const int W_out,
    const int K, const int stride,
    const int padding, const int dilation,
    const int groups,
    const float add_value,
    const float multiply_value)
{
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_out = N * C_out * H_out * W_out;
    if (out_idx >= total_out) return;

    // ---- decode output index -------------------------------------------------
    const int n = out_idx / (C_out * H_out * W_out);
    int rem = out_idx % (C_out * H_out * W_out);
    const int oc = rem / (H_out * W_out);
    rem = rem % (H_out * W_out);
    const int oh = rem / W_out;
    const int ow = rem % W_out;

    // ---- transposed convolution (naïve) --------------------------------------
    float sum = 0.0f;
    if (bias) sum = bias[oc];

    const int out_ch_per_group = C_out / groups;
    const int in_ch_per_group  = C_in  / groups;
    const int g = oc / out_ch_per_group;               // group index

    for (int ic = 0; ic < in_ch_per_group; ++ic) {
        const int ic_global = g * in_ch_per_group + ic;

        // weight base for this (ic_global, oc)
        const int w_base = ((ic_global * C_out + oc) * K);

        #pragma unroll
        for (int kh = 0; kh < K; ++kh) {
            const int ih = (oh + padding - kh * dilation);
            if (ih < 0) continue;
            if (ih % stride != 0) continue;
            const int ihi = ih / stride;
            if (ihi >= H_in) continue;

            #pragma unroll
            for (int kw = 0; kw < K; ++kw) {
                const int iw = (ow + padding - kw * dilation);
                if (iw < 0) continue;
                if (iw % stride != 0) continue;
                const int iwi = iw / stride;
                if (iwi >= W_in) continue;

                const int w_idx = (w_base + kh) * K + kw;
                const float w = weight[w_idx];
                const float inp = input[((n * C_in + ic_global) * H_in + ihi) * W_in + iwi];
                sum += inp * w;
            }
        }
    }

    // ---- point‑wise fused ops -----------------------------------------------
    sum += add_value;
    sum = fminf(sum, 0.0f);                 // clamp to 0 if positive
    float out_val = fast_gelu(sum);
    out_val *= multiply_value;

    output[out_idx] = out_val;
}

// Host wrapper that launches the kernel
void deconv_fused(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    const int N, const int C_in, const int C_out,
    const int H_in, const int W_in,
    const int H_out, const int W_out,
    const int K, const int stride,
    const int padding, const int dilation,
    const int groups,
    const float add_value,
    const float multiply_value)
{
    const int block = 256;
    const int total_out = N * C_out * H_out * W_out;
    const int grid = (total_out + block - 1) / block;

    const float* bias_ptr = bias.numel() ? bias.data_ptr<float>() : nullptr;

    deconv_fused_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, C_out, H_in, W_in, H_out, W_out,
        K, stride, padding, dilation, groups,
        add_value, multiply_value);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
}
"""

# ----------------------------------------------------------------------
# Inline C++ source – pybind11 binding
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void deconv_fused(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    const int N, const int C_in, const int C_out,
    const int H_in, const int W_in,
    const int H_out, const int W_out,
    const int K, const int stride,
    const int padding, const int dilation,
    const int groups,
    const float add_value,
    const float multiply_value);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("deconv_fused", &deconv_fused, "Fused deconvolution + pointwise ops");
}
"""

# ----------------------------------------------------------------------
# Compile the extension (cached)
# ----------------------------------------------------------------------
_fused_module = None

def _get_fused_module():
    global _fused_module
    if _fused_module is None:
        _fused_module = load_inline(
            name="fused_deconv",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            with_cuda=True,
        )
    return _fused_module

# ----------------------------------------------------------------------
# The functional model – replaces the original five‑kernel version
# ----------------------------------------------------------------------
def functional_model(
    x: torch.Tensor,
    *,
    conv_transpose_weight: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
    conv_transpose_stride: int,
    conv_transpose_padding: int,
    conv_transpose_output_padding: int,
    conv_transpose_groups: int,
    conv_transpose_dilation: int,
    add_value: float,
    multiply_value: float,
) -> torch.Tensor:
    # Ensure contiguous memory layout (required by the custom kernel)
    x = x.contiguous()
    weight = conv_transpose_weight.contiguous()
    bias = conv_transpose_bias.contiguous() if conv_transpose_bias is not None else torch.zeros(1, device=x.device)

    # Geometry of the transposed convolution
    stride = conv_transpose_stride
    padding = conv_transpose_padding
    output_padding = conv_transpose_output_padding
    dilation = conv_transpose_dilation
    K = weight.shape[2]                     # square kernel assumed

    N, C_in, H_in, W_in = x.shape
    C_out = weight.shape[1]

    H_out = (H_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1

    # Allocate output tensor
    out = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)

    # Call the fused CUDA kernel
    fused = _get_fused_module()
    fused.deconv_fused(
        x, weight, bias, out,
        N, C_in, C_out, H_in, W_in, H_out, W_out,
        K, stride, padding, dilation, conv_transpose_groups,
        add_value, multiply_value,
    )
    return out

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
