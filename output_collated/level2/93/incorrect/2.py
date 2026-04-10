# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_152044/code_4.py
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

# --- CUDA Kernel Code ---
# Note: This implementation uses a direct accumulation method for ConvTranspose2d.
# In a high-performance scenario, one might use shared memory or cuDNN-like strategies,
# but for the specified requirements, this fused kernel eliminates global memory stalls.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float add_value,
    float multiply_value,
    int B, int C_in, int C_out, 
    int H_in, int W_in, 
    int H_out, int W_out,
    int K, int stride, int padding
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= B * C_out * H_out * W_out) return;

    int temp = out_idx;
    int w_out = temp % W_out; temp /= W_out;
    int h_out = temp % H_out; temp /= H_out;
    int c_out = temp % C_out; temp /= C_out;
    int b = temp;

    float val = bias[c_out];

    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int h_in = (h_out + padding - kh);
                int w_in = (w_out + padding - kw);
                
                if (h_in >= 0 && h_in < H_in * stride && h_in % stride == 0 &&
                    w_in >= 0 && w_in < W_in * stride && w_in % stride == 0) {
                    
                    int h_idx = h_in / stride;
                    int w_idx = w_in / stride;
                    
                    int in_idx = ((b * C_in + c_in) * H_in + h_idx) * W_in + w_idx;
                    int w_idx_flat = ((c_out * C_in + c_in) * K + kh) * K + kw;
                    
                    val += input[in_idx] * weight[w_idx_flat];
                }
            }
        }
    }

    val += add_value;
    val = fminf(val, 0.0f);
    val = 0.5f * val * (1.0f + erff(val * 0.7071067811865476f));
    output[out_idx] = val * multiply_value;
}

void fused_op_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output, float add_val, float mult_val,
    int k, int s, int p
) {
    int B = input.size(0); int C_in = input.size(1);
    int H_in = input.size(2); int W_in = input.size(3);
    int C_out = weight.size(1);
    int H_out = output.size(2); int W_out = output.size(3);
    
    int total = B * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    fused_conv_transpose_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), add_val, mult_val,
        B, C_in, C_out, H_in, W_in, H_out, W_out, k, s, p
    );
}
"""

cpp_source = """
void fused_op_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, float, int, int, int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward);
}
"""

module = load_inline(name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_kernel, with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, add_value, multiply_value):
    k = conv_transpose_weight.size(2)
    h_out = (x.size(2) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + k + conv_transpose_output_padding
    w_out = (x.size(3) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + k + conv_transpose_output_padding
    output = torch.empty((x.size(0), conv_transpose_weight.size(1), h_out, w_out), device=x.device)
    module.fused_op(x, conv_transpose_weight, conv_transpose_bias, output, add_value, multiply_value, k, conv_transpose_stride, conv_transpose_padding)
    return output
