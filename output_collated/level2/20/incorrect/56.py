# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_133559/code_27.py
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

# Optimized monolithic CUDA kernel that performs transpose convolution 
# accumulation and the f(x) = x * (2*x + bias + 1) element-wise operation 
# in a single pass to maximize L1 cache usage and minimize global memory traffic.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose_postproc_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_d, const int in_h, const int in_w,
    const int out_d, const int out_h, const int out_w,
    const int kd, const int kh, const int kw,
    const int sd, const int sh, const int sw,
    const int pd, const int ph, const int pw,
    const int groups
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch_size * out_channels * out_d * out_h * out_w) return;

    int spatial_size = out_d * out_h * out_w;
    int b = out_idx / (out_channels * spatial_size);
    int c = (out_idx / spatial_size) % out_channels;
    int od = (out_idx / (out_h * out_w)) % out_d;
    int oh = (out_idx / out_w) % out_h;
    int ow = out_idx % out_w;

    float acc = 0.0f;
    int group_id = c / (out_channels / groups);
    int in_c_start = group_id * (in_channels / groups);
    int in_c_end = in_c_start + (in_channels / groups);

    // Stride-based Transpose Conv logic: Map output coord back to input
    for (int ic = in_c_start; ic < in_c_end; ++ic) {
        for (int z = 0; z < kd; ++z) {
            int id = od - pd + z;
            if (id % sd != 0 || (id /= sd) < 0 || id >= in_d) continue;
            for (int y = 0; y < kh; ++y) {
                int ih = oh - ph + y;
                if (ih % sh != 0 || (ih /= sh) < 0 || ih >= in_h) continue;
                for (int x = 0; x < kw; ++x) {
                    int iw = ow - pw + x;
                    if (iw % sw != 0 || (iw /= sw) < 0 || iw >= in_w) continue;
                    
                    int i_idx = ((b * in_channels + ic) * in_d + id) * (in_h * in_w) + ih * in_w + iw;
                    int w_idx = ((c * (in_channels / groups) + (ic - in_c_start)) * kd + z) * (kh * kw) + y * kw + x;
                    acc += input[i_idx] * weight[w_idx];
                }
            }
        }
    }

    float b_val = bias[c];
    float x = acc;
    // Fused post-processing: out = x * (2*x + bias + 1)
    output[out_idx] = x * (2.0f * x + b_val + 1.0f);
}

void fused_conv_transpose_postproc_forward(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
    torch::Tensor& output, const int sd, const int sh, const int sw,
    const int pd, const int ph, const int pw, const int groups
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(1) * groups;
    int in_d = input.size(2), in_h = input.size(3), in_w = input.size(4);
    int out_d = output.size(2), out_h = output.size(3), out_w = output.size(4);
    int kd = weight.size(2), kh = weight.size(3), kw = weight.size(4);

    int total_elements = batch_size * out_channels * out_d * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_conv_transpose_postproc_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels, in_d, in_h, in_w, out_d, out_h, out_w,
        kd, kh, kw, sd, sh, sw, pd, ph, pw, groups
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_conv_transpose_postproc_forward(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, torch::Tensor&, int, int, int, int, int, int, int);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_postproc", &fused_conv_transpose_postproc_forward, "Fused Transpose Conv + Post Processing");
}
"""

fused_ext = load_inline(
    name='fused_conv_transpose_postproc_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias):
    # Calculate output shape
    b, in_c, in_d, in_h, in_w = x.shape
    kd, kh, kw = conv_transpose_weight.shape[2:]
    sd, sh, sw = conv_transpose_stride
    pd, ph, pw = conv_transpose_padding
    opd, oph, opw = conv_transpose_output_padding
    out_c = conv_transpose_weight.shape[1] * conv_transpose_groups
    
    out_d = (in_d - 1) * sd - 2 * pd + kd + opd
    out_h = (in_h - 1) * sh - 2 * ph + kh + oph
    out_w = (in_w - 1) * sw - 2 * pw + kw + opw
    
    output = torch.empty((b, out_c, out_d, out_h, out_w), device=x.device, dtype=x.dtype)
    fused_ext.fused_conv_transpose_postproc(x, conv_transpose_weight, bias.view(-1), output, 
                                            sd, sh, sw, pd, ph, pw, conv_transpose_groups)
    return output
