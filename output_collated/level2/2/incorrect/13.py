# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_163027/code_4.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape', 'scaling_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias', 'scaling_factor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

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
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
    if 'scaling_factor' in flat_state:
        state_kwargs['scaling_factor'] = flat_state['scaling_factor']
    else:
        state_kwargs['scaling_factor'] = getattr(model, 'scaling_factor')
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

# CUDA kernel for fused operation
# The manual convolution implementation uses a direct approach suitable for 
# mid-sized kernels. Fusing bias, clamps, and scaling eliminates intermediate loads/stores.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ add_bias,
    float* __restrict__ output,
    int batch_size, int in_c, int out_c,
    int in_h, int in_w, int out_h, int out_w,
    int k_s, int stride, int padding, float scaling_factor
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_c * out_h * out_w;
    if (out_idx >= total) return;

    int tmp = out_idx;
    int w_out = tmp % out_w; tmp /= out_w;
    int h_out = tmp % out_h; tmp /= out_h;
    int c_out = tmp % out_c; tmp /= out_c;
    int b = tmp;

    float sum = 0.0f;

    // Direct loop over input positions that contribute to this output pixel
    for (int ky = 0; ky < k_s; ++ky) {
        for (int kx = 0; kx < k_s; ++kx) {
            int h_in_pos = h_out + padding - ky;
            int w_in_pos = w_out + padding - kx;

            if (h_in_pos % stride == 0 && w_in_pos % stride == 0) {
                int h_in = h_in_pos / stride;
                int w_in = w_in_pos / stride;

                if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                    for (int c_in = 0; c_in < in_c; ++c_in) {
                        float val = input[((b * in_c + c_in) * in_h + h_in) * in_w + w_in];
                        float w = weight[((c_out * in_c + c_in) * k_s + ky) * k_s + kx];
                        sum += val * w;
                    }
                }
            }
        }
    }

    sum += conv_bias[c_out] + add_bias[c_out];
    sum = fmaxf(0.0f, fminf(1.0f, sum));
    sum *= scaling_factor;
    sum = fmaxf(0.0f, fminf(1.0f, sum));
    sum /= scaling_factor;

    output[out_idx] = sum;
}

void fused_conv_transpose_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor conv_bias, 
    torch::Tensor add_bias, torch::Tensor output,
    int stride, int padding, float scaling_factor
) {
    int b = output.size(0), c = output.size(1), h = output.size(2), w = output.size(3);
    int in_c = input.size(1), in_h = input.size(2), in_w = input.size(3);
    int k_s = weight.size(2);
    
    int total = b * c * h * w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fused_conv_transpose_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), conv_bias.data_ptr<float>(),
        add_bias.data_ptr<float>(), output.data_ptr<float>(),
        b, in_c, c, in_h, in_w, h, w, k_s, stride, padding, scaling_factor
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_conv_transpose_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor conv_bias, torch::Tensor add_bias, torch::Tensor output, int stride, int padding, float scaling_factor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose", &fused_conv_transpose_forward, "Fused kernel");
}
"""

fused_ext = load_inline(name='fused_ops', cpp_sources=cpp_source, cuda_sources=cuda_kernel, 
                       extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias, scaling_factor):
    b, in_c, in_h, in_w = x.shape
    out_c, _, k_h, k_w = conv_transpose_weight.shape
    out_h = (in_h - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + k_h + conv_transpose_output_padding[0]
    out_w = (in_w - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + k_w + conv_transpose_output_padding[1]
    
    output = torch.empty(b, out_c, out_h, out_w, device=x.device, dtype=x.dtype)
    fused_ext.fused_conv_transpose(x, conv_transpose_weight, conv_transpose_bias, bias.squeeze(), 
                                   output, conv_transpose_stride[0], conv_transpose_padding[0], float(scaling_factor))
    return output
