# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_163518/code_4.py
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

# CUDA kernel with fused deconvolution and post-processing
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void fused_op_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int C_out, int H, int W,
    int K, int S, int P, int OP,
    int H_out, int W_out,
    float scaling_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C_out * H_out * W_out;
    if (idx >= total_elements) return;

    int tmp = idx;
    int w_out = tmp % W_out; tmp /= W_out;
    int h_out = tmp % H_out; tmp /= H_out;
    int c_out = tmp % C_out; tmp /= C_out;
    int b = tmp;

    float val = bias[c_out];

    // Naive implementation of Transposed Convolution: O(C_in * K * K * B * H_out * W_out)
    // Optimized for GPU memory access patterns
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int h_in_pos = h_out + P - kh;
                int w_in_pos = w_out + P - kw;
                
                if (h_in_pos % S == 0 && w_in_pos % S == 0) {
                    int h_in = h_in_pos / S;
                    int w_in = w_in_pos / S;
                    
                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        int in_idx = ((b * C_in + c_in) * H + h_in) * W + w_in;
                        int w_idx = ((c_in * C_out + c_out) * K + kh) * K + kw;
                        val += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
    }

    // Fused element-wise post-processing
    val = fmaxf(0.0f, fminf(1.0f, val));
    val *= scaling_factor;
    val = fmaxf(0.0f, fminf(1.0f, val));
    val /= scaling_factor;

    output[idx] = val;
}

void fused_op_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output, int S, int P, int OP, float scaling_factor
) {
    int B = input.size(0);
    int C_in = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int C_out = weight.size(1);
    int K = weight.size(2);
    int H_out = output.size(2);
    int W_out = output.size(3);

    int total_elements = B * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_op_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), B, C_in, C_out, H, W,
        K, S, P, OP, H_out, W_out, scaling_factor
    );
}
"""

cpp_source = """
void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int S, int P, int OP, float scaling_factor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused deconv and activation");
}
"""

fused_lib = load_inline(name='fused_lib', cpp_sources=cpp_source, cuda_sources=cuda_kernel, 
                       extra_cuda_cflags=['-O3'], with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias, scaling_factor):
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = conv_transpose_weight.contiguous()
    
    B, C_in, H, W = x.shape
    C_out, _, K, _ = weight.shape
    H_out = (H - 1) * conv_transpose_stride - 2 * conv_transpose_padding + K + conv_transpose_output_padding
    W_out = (W - 1) * conv_transpose_stride - 2 * conv_transpose_padding + K + conv_transpose_output_padding
    
    output = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    fused_lib.fused_op(x, weight, bias.squeeze(), output, 
                       conv_transpose_stride, conv_transpose_padding, 
                       conv_transpose_output_padding, scaling_factor)
    return output
