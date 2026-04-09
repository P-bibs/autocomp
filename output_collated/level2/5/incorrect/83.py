# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_123254/code_22.py
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

# Optimized CUDA Kernel: Manual Transposed Conv2D + Bias + Tanh
# Implementing Grid-Stride loops for high occupancy and performance
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void conv_transpose_fused_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    int N, int in_c, int in_h, int in_w,
    int out_c, int k_h, int k_w,
    int stride, int padding, 
    int out_h, int out_w) {

    int total_elements = N * out_c * out_h * out_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_all = blockDim.x * gridDim.x;

    for (int i = idx; i < total_elements; i += stride_all) {
        int n = i / (out_c * out_h * out_w);
        int oc = (i / (out_h * out_w)) % out_c;
        int oh = (i / out_w) % out_h;
        int ow = i % out_w;

        float sum = 0.0f;
        // Transposed convolution index logic
        // We look for contributions from the input pixels to the current output pixel
        for (int ic = 0; ic < in_c; ++ic) {
            for (int kh = 0; kh < k_h; ++kh) {
                for (int kw = 0; kw < k_w; ++kw) {
                    int ih_idx = oh + padding - kh;
                    int iw_idx = ow + padding - kw;
                    
                    if (ih_idx % stride == 0 && iw_idx % stride == 0) {
                        int ih = ih_idx / stride;
                        int iw = iw_idx / stride;
                        
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            float in_val = input[((n * in_c + ic) * in_h + ih) * in_w + iw];
                            float w_val = weight[((ic * out_c + oc) * k_h + kh) * k_w + kw];
                            sum += in_val * w_val;
                        }
                    }
                }
            }
        }
        output[i] = tanhf(sum - bias[oc]);
    }
}

void fused_op_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int padding) {
    int N = x.size(0), in_c = x.size(1), in_h = x.size(2), in_w = x.size(3);
    int out_c = weight.size(1);
    int k_h = weight.size(2), k_w = weight.size(3);
    int out_h = output.size(2);
    int out_w = output.size(3);
    
    int total_elements = N * out_c * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    
    conv_transpose_fused_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        N, in_c, in_h, in_w, out_c, k_h, k_w, stride, padding, out_h, out_w
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_cuda, "Fused Transposed Conv + Bias + Tanh");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, conv_transpose_dilation, bias):
    # Ensure memory is contiguous
    x = x.contiguous()
    conv_transpose_weight = conv_transpose_weight.contiguous()
    
    N, in_c, in_h, in_w = x.shape
    out_c = conv_transpose_weight.shape[1]
    k_h, k_w = conv_transpose_weight.shape[2:]
    
    out_h = (in_h - 1) * conv_transpose_stride - 2 * conv_transpose_padding + k_h + conv_transpose_output_padding
    out_w = (in_w - 1) * conv_transpose_stride - 2 * conv_transpose_padding + k_w + conv_transpose_output_padding
    
    output = torch.zeros((N, out_c, out_h, out_w), device='cuda')
    
    # Run custom kernel
    fused_ext.fused_op(x, conv_transpose_weight, bias.view(-1), output, conv_transpose_stride, conv_transpose_padding)
    return output
