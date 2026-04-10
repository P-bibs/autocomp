# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_160727/code_12.py
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

# ============================================================================
# CUDA Kernel Implementation
# The kernel implements a deconvolution (ConvTranspose2d) essentially as a 
# Gather operation where each output pixel accumulates contributions from 
# overlapping kernel windows.
# ============================================================================

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_transpose_kernel(
    const float* __restrict__ input,      
    const float* __restrict__ weight,     
    const float* __restrict__ bias,       
    float* __restrict__ output,           
    float add_val, float mul_val,
    int batch_size, int in_channels, int out_channels,
    int in_h, int in_w, int out_h, int out_w,
    int k_h, int k_w, int stride, int padding
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch_size * out_channels * out_h * out_w) return;

    // Map linear index to 4D coordinates
    int tmp = out_idx;
    int x = tmp % out_w; tmp /= out_w;
    int y = tmp % out_h; tmp /= out_h;
    int oc = tmp % out_channels; tmp /= out_channels;
    int b = tmp;

    float acc = (bias != nullptr) ? bias[oc] : 0.0f;

    // For ConvTranspose, output location (y, x) is affected by input (iy, ix) 
    // where iy*stride + ky - padding == y => iy = (y + padding - ky) / stride
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int ky = 0; ky < k_h; ++ky) {
            for (int kx = 0; kx < k_w; ++kx) {
                int iy_off = y + padding - ky;
                int ix_off = x + padding - kx;
                
                if (iy_off % stride == 0 && ix_off % stride == 0) {
                    int iy = iy_off / stride;
                    int ix = ix_off / stride;

                    if (iy >= 0 && iy < in_h && ix >= 0 && ix < in_w) {
                        int in_idx = ((b * in_channels + ic) * in_h + iy) * in_w + ix;
                        int w_idx = ((ic * out_channels + oc) * k_h + ky) * k_w + kx;
                        acc += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
    }

    // Fused ops
    acc += add_val;
    acc = fminf(acc, 0.0f);
    acc = fast_gelu(acc);
    output[out_idx] = acc * mul_val;
}

void fused_conv_transpose_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    float add_val, float mul_val, int stride, int padding
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_channels = weight.size(1);
    int k_h = weight.size(2);
    int k_w = weight.size(3);
    int out_h = output.size(2);
    int out_w = output.size(3);

    int total_elements = batch_size * out_channels * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_conv_transpose_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.data_ptr<float>(), output.data_ptr<float>(),
        add_val, mul_val, batch_size, in_channels, out_channels,
        in_h, in_w, out_h, out_w, k_h, k_w, stride, padding
    );
}
"""

cpp_source = """
#include <torch/extension.h>
void fused_conv_transpose_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, float add_val, float mul_val, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose", &fused_conv_transpose_forward, "Fused ConvTranspose2d + Ops");
}
"""

fused_ext = load_inline(
    name='fused_conv_transpose',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

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
    # Output dim calculation: (H_in - 1)*stride - 2*pad + dilation*(k-1) + out_pad + 1
    # Assuming standard parameters for the kernel-supported logic
    h_out = (x.size(2) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_weight.size(2) + conv_transpose_output_padding
    w_out = (x.size(3) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_weight.size(3) + conv_transpose_output_padding
    
    out = torch.empty((x.size(0), conv_transpose_weight.size(1), h_out, w_out), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_conv_transpose(
        x, conv_transpose_weight, conv_transpose_bias, out,
        float(add_value), float(multiply_value),
        int(conv_transpose_stride), int(conv_transpose_padding)
    )
    return out
