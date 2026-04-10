# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_153448/code_14.py
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
# Optimized CUDA kernel: Replaces F.conv_transpose2d + element-wise chain
# This eliminates the massive intermediate global memory allocation.
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_transpose_kernel(
    const float* __restrict__ input,      // (N, Cin, Hin, Win)
    const float* __restrict__ weight,     // (Cin, Cout, K, K) - Note: standard conv_transpose weight layout
    const float* __restrict__ bias,
    float* __restrict__ output,           // (N, Cout, Hout, Wout)
    const int N, const int Cin, const int Hin, const int Win,
    const int Cout, const int K,
    const int stride, const int padding,
    const float add_val, const float mul_val,
    const int Hout, const int Wout)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * Cout * Hout * Wout) return;

    int tmp = idx;
    int ow = tmp % Wout; tmp /= Wout;
    int oh = tmp % Hout; tmp /= Hout;
    int oc = tmp % Cout; tmp /= Cout;
    int n  = tmp;

    float sum = (bias != nullptr) ? bias[oc] : 0.0f;

    // Transposed Convolution logic:
    // Output element (oc, oh, ow) is affected by input (ci, ih, iw) if:
    // oh = ih * stride + kh - padding
    // iw = iw * stride + kw - padding
    // Solving for ih, iw:
    // ih = (oh + padding - kh) / stride
    // iw = (ow + padding - kw) / stride
    
    for (int ci = 0; ci < Cin; ++ci) {
        for (int kh = 0; kh < K; ++kh) {
            int ih_idx = oh + padding - kh;
            if (ih_idx < 0 || ih_idx % stride != 0) continue;
            int ih = ih_idx / stride;
            if (ih >= Hin) continue;

            for (int kw = 0; kw < K; ++kw) {
                int iw_idx = ow + padding - kw;
                if (iw_idx < 0 || iw_idx % stride != 0) continue;
                int iw = iw_idx / stride;
                if (iw >= Win) continue;

                float in_val = input[((n * Cin + ci) * Hin + ih) * Win + iw];
                // Weight layout for conv_transpose2d: (Cin, Cout, K, K)
                float w_val = weight[((ci * Cout + oc) * K + kh) * K + kw];
                sum += in_val * w_val;
            }
        }
    }

    // Fused element-wise operations
    float val = sum + add_val;
    val = fminf(val, 0.0f);
    val = fast_gelu(val);
    output[idx] = val * mul_val;
}

void fused_conv_transpose_launch(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride, int padding, float add_val, float mul_val) 
{
    int N = input.size(0); int Cin = input.size(1);
    int Hin = input.size(2); int Win = input.size(3);
    int Cout = weight.size(1);
    int K = weight.size(2);
    int Hout = output.size(2); int Wout = output.size(3);

    int total_elements = N * Cout * Hout * Wout;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_conv_transpose_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, Cin, Hin, Win, Cout, K, stride, padding,
        add_val, mul_val, Hout, Wout
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_conv_transpose_launch(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int padding, float add_val, float mul_val);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose", &fused_conv_transpose_launch, "Fused ConvTranspose2d + Elemwise");
}
"""

module = load_inline(
    name='fused_conv_transpose',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, add_value, multiply_value):
    # Calculate output dimensions
    N, Cin, Hin, Win = x.shape
    Cout, _, K, _ = conv_transpose_weight.shape
    Hout = (Hin - 1) * conv_transpose_stride - 2 * conv_transpose_padding + K + conv_transpose_output_padding
    Wout = (Win - 1) * conv_transpose_stride - 2 * conv_transpose_padding + K + conv_transpose_output_padding
    
    out = torch.empty((N, Cout, Hout, Wout), device=x.device, dtype=x.dtype)
    module.fused_conv_transpose(x, conv_transpose_weight, conv_transpose_bias, out, 
                                conv_transpose_stride, conv_transpose_padding, 
                                float(add_value), float(multiply_value))
    return out
