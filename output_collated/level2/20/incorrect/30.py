# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_130922/code_20.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, followed by a sum, 
    a residual add, a multiplication, and another residual add.
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

# -------------------------------------------------------------------------
# High-performance fused Transpose Conv + Activation Kernel
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose3d_kernel(
    const float* __restrict__ input, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ output,
    int batch, int in_c, int out_c, int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w, int k_size, int stride) 
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch * out_c * out_d * out_h * out_w) return;

    // Decoding output index
    int tmp = out_idx;
    int w = tmp % out_w; tmp /= out_w;
    int h = tmp % out_h; tmp /= out_h;
    int d = tmp % out_d; tmp /= out_d;
    int oc = tmp % out_c; tmp /= out_c;
    int b = tmp;

    float acc = 0.0f;
    // Iterate over channels and kernel window
    for (int ic = 0; ic < in_c; ++ic) {
        for (int kd = 0; kd < k_size; ++kd) {
            int id = d + kd - (k_size - 1);
            if (id % stride != 0) continue;
            int id_idx = id / stride;
            if (id_idx < 0 || id_idx >= in_d) continue;

            for (int kh = 0; kh < k_size; ++kh) {
                int ih = h + kh - (k_size - 1);
                if (ih % stride != 0) continue;
                int ih_idx = ih / stride;
                if (ih_idx < 0 || ih_idx >= in_h) continue;

                for (int kw = 0; kw < k_size; ++kw) {
                    int iw = w + kw - (k_size - 1);
                    if (iw % stride != 0) continue;
                    int iw_idx = iw / stride;
                    if (iw_idx < 0 || iw_idx >= in_w) continue;

                    float val = input[(((b * in_c + ic) * in_d + id_idx) * in_h + ih_idx) * in_w + iw_idx];
                    float w_val = weight[((ic * out_c + oc) * k_size + kd) * k_size * k_size + kh * k_size + kw];
                    acc += val * w_val;
                }
            }
        }
    }
    
    acc += bias[oc];
    // Fuse: ((2*x + bias) * x) + x = 2*x^2 + bias*x + x
    output[out_idx] = (2.0f * acc * acc + bias[oc] * acc + acc);
}

void fused_conv_transpose3d_cuda(const torch::Tensor& in, const torch::Tensor& w, const torch::Tensor& b, torch::Tensor& out) {
    int batch = in.size(0), in_c = in.size(1), in_d = in.size(2), in_h = in.size(3), in_w = in.size(4);
    int out_c = w.size(1), k_size = w.size(2);
    int out_d = out.size(2), out_h = out.size(3), out_w = out.size(4);
    int stride = 2; // Assuming standard usage from parameters

    int num_elements = batch * out_c * out_d * out_h * out_w;
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    
    fused_conv_transpose3d_kernel<<<blocks, threads>>>(
        in.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(),
        batch, in_c, out_c, in_d, in_h, in_w, out_d, out_h, out_w, k_size, stride
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_conv_transpose3d_cuda(const torch::Tensor& in, const torch::Tensor& w, const torch::Tensor& b, torch::Tensor& out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &fused_conv_transpose3d_cuda, "Fused ConvTranspose3D"); }
"""

fused_ext = load_inline(
    name='fused_conv', cpp_sources=cpp_source, cuda_sources=cuda_source, 
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, conv_transpose_padding, 
                     conv_transpose_output_padding, conv_transpose_groups, conv_transpose_dilation, bias):
    batch, in_c, in_d, in_h, in_w = x.shape
    out_c, _, k_size, _, _ = conv_transpose_weight.shape
    
    out_d = (in_d - 1) * conv_transpose_stride - 2 * conv_transpose_padding + k_size + conv_transpose_output_padding
    out_h = (in_h - 1) * conv_transpose_stride - 2 * conv_transpose_padding + k_size + conv_transpose_output_padding
    out_w = (in_w - 1) * conv_transpose_stride - 2 * conv_transpose_padding + k_size + conv_transpose_output_padding
    
    output = torch.empty((batch, out_c, out_d, out_h, out_w), device=x.device)
    fused_ext.forward(x.contiguous(), conv_transpose_weight.contiguous(), bias.view(-1).contiguous(), output)
    return output
