# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_045050/code_20.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x', 'add_input']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

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
    # State for conv_transpose (nn.ConvTranspose3d)
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

# CUDA kernel source
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose_add_hardswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride, int padding, int output_padding,
    int groups
) {
    int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_d * out_h * out_w;
    if (g_idx >= total_elements) return;

    int tmp = g_idx;
    int ow = tmp % out_w; tmp /= out_w;
    int oh = tmp % out_h; tmp /= out_h;
    int od = tmp % out_d; tmp /= out_d;
    int oc = tmp % out_channels; tmp /= out_channels;
    int b  = tmp;

    int oc_per_group = out_channels / groups;
    int ic_per_group = in_channels / groups;
    int group_idx = oc / oc_per_group;
    
    float val = (bias != nullptr) ? bias[oc] : 0.0f;

    // Standard Transposed Conv logic: Kernel acts on output space 
    // mapping to input space via stride/padding inversion
    for (int kd = 0; kd < kernel_d; ++kd) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int id_raw = (od + padding - kd);
                int ih_raw = (oh + padding - kh);
                int iw_raw = (ow + padding - kw);

                if (id_raw % stride == 0 && ih_raw % stride == 0 && iw_raw % stride == 0) {
                    int id = id_raw / stride;
                    int ih = ih_raw / stride;
                    int iw = iw_raw / stride;

                    if (id >= 0 && id < in_d && ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        for (int ic_g = 0; ic_g < ic_per_group; ++ic_g) {
                            int ic = group_idx * ic_per_group + ic_g;
                            int in_idx = (((b * in_channels + ic) * in_d + id) * in_h + ih) * in_w + iw;
                            int w_idx = (((oc * ic_per_group + ic_g) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
                            val += input[in_idx] * weight[w_idx];
                        }
                    }
                }
            }
        }
    }

    val += add_input[g_idx];
    // hardswish(x) = x * relu6(x + 3) / 6
    float hswish = val * fminf(fmaxf(val + 3.0f, 0.0f), 6.0f) / 6.0f;
    output[g_idx] = val * hswish;
}

void launch_kernel(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor add_input, torch::Tensor output,
    int stride, int padding, int output_padding, int groups
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(1) * groups;
    int in_d = input.size(2), in_h = input.size(3), in_w = input.size(4);
    int out_d = output.size(2), out_h = output.size(3), out_w = output.size(4);
    int kernel_d = weight.size(2), kernel_h = weight.size(3), kernel_w = weight.size(4);

    int total_elements = batch_size * out_channels * out_d * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_conv_transpose_add_hardswish_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.data_ptr<float>(), add_input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w, out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride, padding, output_padding, groups
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_kernel(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                  torch::Tensor add_input, torch::Tensor output,
                  int stride, int padding, int output_padding, int groups);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_kernel, "Fused ConvTranspose3d + Add + Hardswish");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, add_input, *, conv_transpose_weight, conv_transpose_bias, 
                     conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding, 
                     conv_transpose_groups, conv_transpose_dilation, bias):
    # Calculate output dims
    D_in, H_in, W_in = x.shape[2:]
    K = conv_transpose_weight.shape[2]
    D_out = (D_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + K + conv_transpose_output_padding
    
    out = torch.empty((x.shape[0], conv_transpose_weight.shape[1] * conv_transpose_groups, D_out, D_out, D_out), 
                      device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, add_input, out, 
                       conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding, 
                       conv_transpose_groups)
    return out
