# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_123254/code_17.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'bias_shape', 'stride', 'padding', 'output_padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
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

# Combined CUDA Kernel for Conv Transpose + Bias Tanh
# We fuse the convolution, bias addition, and tanh into a single kernel to maximize register reuse 
# and minimize global memory round-trips.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_conv_tr_bias_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int in_channels, int in_H, int in_W,
    int out_channels, int kernel_size, int stride, int padding,
    int out_H, int out_W
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * out_channels * out_H * out_W;
    
    if (tid >= total_elements) return;

    // Decoding 1D index to N, C, H, W
    int temp = tid;
    int w_out = temp % out_W; temp /= out_W;
    int h_out = temp % out_H; temp /= out_H;
    int c_out = temp % out_channels; temp /= out_channels;
    int n = temp;

    float sum = 0.0f;
    
    // Perform convolution transpose
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = (h_out + padding - kh);
                int w_in = (w_out + padding - kw);
                
                if (h_in % stride == 0 && w_in % stride == 0) {
                    h_in /= stride;
                    w_in /= stride;
                    
                    if (h_in >= 0 && h_in < in_H && w_in >= 0 && w_in < in_W) {
                        float val = input[((n * in_channels + c_in) * in_H + h_in) * in_W + w_in];
                        float w = weight[((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw];
                        sum += val * w;
                    }
                }
            }
        }
    }
    
    // Apply bias and tanh
    output[tid] = tanhf(sum + bias[c_out]);
}

void fused_op_forward(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride, int padding, int output_padding
) {
    int N = x.size(0);
    int in_channels = x.size(1);
    int in_H = x.size(2);
    int in_W = x.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int out_H = (in_H - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_W = (in_W - 1) * stride - 2 * padding + kernel_size + output_padding;

    int total_elements = N * out_channels * out_H * out_W;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_conv_tr_bias_tanh_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        N, in_channels, in_H, in_W, out_channels, kernel_size, stride, padding, out_H, out_W
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int padding, int output_padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op_forward", &fused_op_forward, "Fused ConvTranspose + Bias + Tanh");
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
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    bias,
):
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_size, _ = conv_transpose_weight.shape
    
    out_height = (in_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    out_width = (in_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    
    output = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Run the fused kernel which avoids writing intermediate convolution results to global memory
    fused_ext.fused_op_forward(
        x, conv_transpose_weight, bias.view(-1), output,
        conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding
    )
    
    return output
