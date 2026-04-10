# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_141042/code_6.py
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
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# CUDA Kernel: Fused Conv2d, Subtraction*2, and Mish
# ------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_H 8
#define TILE_W 8

__device__ __forceinline__ float mish(float x) {
    // x * tanh(softplus(x)) where softplus(x) = log(1 + exp(x))
    // Use __logf and __expf for performance via fast-math
    float sp = __logf(__expf(x) + 1.0f);
    return x * tanhf(sp);
}

template <typename scalar_t>
__global__ void fused_conv_mish_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int N, const int Cin, const int Cout,
    const int Hin, const int Win, const int Hout, const int Wout,
    const int Kh, const int Kw,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const float sub1, const float sub2) 
{
    const int n = blockIdx.z / Cout;
    const int cout = blockIdx.z % Cout;
    const int out_x = blockIdx.x * TILE_W + threadIdx.x;
    const int out_y = blockIdx.y * TILE_H + threadIdx.y;

    if (out_x >= Wout || out_y >= Hout) return;

    float acc = (bias != nullptr) ? (float)bias[cout] : 0.0f;

    // Convolution logic
    const int in_y_start = out_y * stride_h - pad_h;
    const int in_x_start = out_x * stride_w - pad_w;

    for (int c = 0; c < Cin; ++c) {
        const scalar_t* in_ptr = input + ((n * Cin + c) * Hin + in_y_start) * Win + in_x_start;
        const scalar_t* w_ptr = weight + (cout * Cin + c) * Kh * Kw;

        for (int kh = 0; kh < Kh; ++kh) {
            int iy = in_y_start + kh;
            if (iy >= 0 && iy < Hin) {
                for (int kw = 0; kw < Kw; ++kw) {
                    int ix = in_x_start + kw;
                    if (ix >= 0 && ix < Win) {
                        acc += (float)(in_ptr + kh * Win + kw)[0] * (float)w_ptr[kh * Kw + kw];
                    }
                }
            }
        }
    }

    // Fused operations
    acc = acc - sub1 - sub2;
    output[(n * Cout + cout) * Hout * Wout + out_y * Wout + out_x] = (scalar_t)mish(acc);
}

void fused_op_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride, int padding, float sub1, float sub2) 
{
    const int N = input.size(0);
    const int Cin = input.size(1);
    const int Hin = input.size(2);
    const int Win = input.size(3);
    const int Cout = weight.size(0);
    const int Kh = weight.size(2);
    const int Kw = weight.size(3);
    const int Hout = output.size(2);
    const int Wout = output.size(3);

    dim3 threads(TILE_W, TILE_H);
    dim3 blocks((Wout + TILE_W - 1) / TILE_W, (Hout + TILE_H - 1) / TILE_H, N * Cout);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_conv_mish", ([&] {
        fused_conv_mish_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            N, Cin, Cout, Hin, Win, Hout, Wout, Kh, Kw,
            stride, stride, padding, padding, sub1, sub2
        );
    }));
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int padding, float sub1, float sub2);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused convolution and activation kernel");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups, subtract_value_1, subtract_value_2):
    N, Cin, Hin, Win = x.shape
    Cout, _, Kh, Kw = conv_weight.shape
    Hout = (Hin + 2 * conv_padding - Kh) // conv_stride + 1
    Wout = (Win + 2 * conv_padding - Kw) // conv_stride + 1
    output = torch.empty((N, Cout, Hout, Wout), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(x, conv_weight, conv_bias, output, conv_stride, conv_padding, subtract_value_1, subtract_value_2)
    return output
