# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_152515/code_5.py
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

# Combined CUDA source: ConvTranspose + Fused Activation
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

// Simple kernel performing ConvTranspose2D logic fused with activation post-processing
// Note: For production, a high-performance tiled GEMM implementation is preferred.
__global__ void fused_conv_transpose_activation_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k, int stride, int padding,
    float add_value, float multiply_value) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * out_c * out_h * out_w) return;

    // Output layout: [N, out_c, out_h, out_w]
    int tmp = idx;
    int ow = tmp % out_w; tmp /= out_w;
    int oh = tmp % out_h; tmp /= out_h;
    int oc = tmp % out_c; tmp /= out_c;
    int n = tmp;

    float acc = bias[oc];
    for (int ic = 0; ic < in_c; ++ic) {
        for (int kh = 0; kh < k; ++kh) {
            for (int kw = 0; kw < k; ++kw) {
                int ih = (oh + padding - kh) / stride;
                int iw = (ow + padding - kw) / stride;
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w && 
                    (oh + padding - kh) % stride == 0 && (ow + padding - kw) % stride == 0) {
                    acc += input[((n * in_c + ic) * in_h + ih) * in_w + iw] * 
                           weight[((ic * out_c + oc) * k + kh) * k + kw];
                }
            }
        }
    }

    // Fused post-processing
    acc += add_value;
    acc = (acc < 0.0f) ? acc : 0.0f;
    output[idx] = gelu(acc) * multiply_value;
}

void fused_conv_transpose_op(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, 
                             torch::Tensor output, int stride, int padding, 
                             float add_value, float multiply_value) {
    int batch = x.size(0); int in_c = x.size(1); int in_h = x.size(2); int in_w = x.size(3);
    int out_c = weight.size(1); int k = weight.size(2);
    int out_h = (in_h - 1) * stride - 2 * padding + k;
    int out_w = (in_w - 1) * stride - 2 * padding + k;
    
    int num_elements = batch * out_c * out_h * out_w;
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    
    fused_conv_transpose_activation_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        batch, in_c, in_h, in_w, out_c, out_h, out_w, k, stride, padding, add_value, multiply_value
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_conv_transpose_op(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, 
                             torch::Tensor output, int stride, int padding, 
                             float add_value, float multiply_value);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_conv_transpose_op, "Fused ConvTranspose + Activation");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_transpose_weight, conv_transpose_bias,
    conv_transpose_stride, conv_transpose_padding,
    conv_transpose_output_padding, conv_transpose_groups,
    conv_transpose_dilation, add_value, multiply_value,
):
    # Calculate output dimensions
    out_h = (x.shape[2] - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_weight.shape[2] + conv_transpose_output_padding[0]
    out_w = (x.shape[3] - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_weight.shape[3] + conv_transpose_output_padding[1]
    
    output = torch.empty((x.shape[0], conv_transpose_weight.shape[1], out_h, out_w), device=x.device)
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, output, 
                       conv_transpose_stride, conv_transpose_padding, add_value, multiply_value)
    return output
