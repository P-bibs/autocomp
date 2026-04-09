# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_121229/code_23.py
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

# Optimized CUDA kernel using a direct compute approach for Transposed Convolution.
# This approach replaces standard im2col + gemm with a fused compute kernel to minimize
# global memory round-trips for the element-wise bias subtraction and Tanh activation.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C, int H, int W, int OC, int K,
    int stride, int padding) {

    int out_H = (H - 1) * stride + K - 2 * padding;
    int out_W = (W - 1) * stride + K - 2 * padding;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_idx >= B * OC * out_H * out_W) return;

    // Coordinate mapping
    int tmp = out_idx;
    int w_out = tmp % out_W; tmp /= out_W;
    int h_out = tmp % out_H; tmp /= out_H;
    int oc = tmp % OC;      tmp /= OC;
    int b = tmp;

    float acc = 0.0f;
    // Iterate over channels and kernel spatial dimensions
    for (int ic = 0; ic < C; ++ic) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int h_in = h_out + padding - kh;
                int w_in = w_out + padding - kw;
                
                // Only contribute to output if the mapping originates from a valid stride position
                if (h_in >= 0 && h_in < (H * stride) && w_in >= 0 && w_in < (W * stride)) {
                    if (h_in % stride == 0 && w_in % stride == 0) {
                        int h_i = h_in / stride;
                        int w_i = w_in / stride;
                        acc += input[((b * C + ic) * H + h_i) * W + w_i] * 
                               weight[((oc * C + ic) * K + kh) * K + kw];
                    }
                }
            }
        }
    }
    
    // Fused: Subtract bias (broadcasted) and apply Tanh
    output[out_idx] = tanhf(acc - bias[oc]);
}

void fused_op(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    int B = input.size(0); int C = input.size(1);
    int H = input.size(2); int W = input.size(3);
    int OC = weight.size(0); int K = weight.size(2);
    int stride = 2; int padding = 1;

    int out_H = (H - 1) * stride + K - 2 * padding;
    int out_W = (W - 1) * stride + K - 2 * padding;
    int total_elements = B * OC * out_H * out_W;
    
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_conv_transpose_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), B, C, H, W, OC, K, stride, padding);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused ConvTranspose2d + Sub + Tanh");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride=2, 
                     conv_transpose_padding=1, conv_transpose_output_padding=0, conv_transpose_groups=1, 
                     conv_transpose_dilation=1, bias):
    B, C, H, W = x.shape
    OC, _, K, _ = conv_transpose_weight.shape
    
    # Calculate output dimensions
    out_H = (H - 1) * conv_transpose_stride + K - 2 * conv_transpose_padding + conv_transpose_output_padding
    out_W = (W - 1) * conv_transpose_stride + K - 2 * conv_transpose_padding + conv_transpose_output_padding
    
    output = torch.empty((B, OC, out_H, out_W), device=x.device)
    
    # Run fused kernel
    fused_ext.fused_op(x, conv_transpose_weight, bias.view(-1), output)
    
    return output
