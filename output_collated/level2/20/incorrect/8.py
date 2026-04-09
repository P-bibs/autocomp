# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_125325/code_10.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_transpose_conv_fused_op_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int K, int stride, int padding
) {
    // Grid-stride loop indexing
    int64_t total_elements = (int64_t)N * C_out * D_out * H_out * W_out;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total_elements; i += grid_size) {
        int64_t tmp = i;
        int w_o = tmp % W_out; tmp /= W_out;
        int h_o = tmp % H_out; tmp /= H_out;
        int d_o = tmp % D_out; tmp /= D_out;
        int c_o = tmp % C_out; tmp /= C_out;
        int n   = tmp;

        float val = 0.0f;
        // Naive transpose conv summation
        for (int c_i = 0; c_i < C_in; ++c_i) {
            for (int kd = 0; kd < K; ++kd) {
                for (int kh = 0; kh < K; ++kh) {
                    for (int kw = 0; kw < K; ++kw) {
                        int d_i = (d_o + padding - kd);
                        int h_i = (h_o + padding - kh);
                        int w_i = (w_o + padding - kw);
                        if (d_i % stride == 0 && h_i % stride == 0 && w_i % stride == 0) {
                            d_i /= stride; h_i /= stride; w_i /= stride;
                            if (d_i >= 0 && d_i < D_in && h_i >= 0 && h_i < H_in && w_i >= 0 && w_i < W_in) {
                                float w_val = weight[(((c_o * C_in + c_i) * K + kd) * K + kh) * K + kw];
                                float i_val = input[(((n * C_in + c_i) * D_in + d_i) * H_in + h_i) * W_in + w_i];
                                val += i_val * w_val;
                            }
                        }
                    }
                }
            }
        }
        val += bias[c_o];
        output[i] = ((val + bias[c_o]) + val) * val + val;
    }
}

void launch_fused_kernel(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, torch::Tensor& output, 
                         int stride, int padding, int K) {
    int N = input.size(0); int C_in = input.size(1);
    int D_in = input.size(2); int H_in = input.size(3); int W_in = input.size(4);
    int C_out = weight.size(1);
    int D_out = output.size(2); int H_out = output.size(3); int W_out = output.size(4);
    
    int threads = 256;
    int blocks = 1024;
    fused_transpose_conv_fused_op_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        N, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out, K, stride, padding
    );
}
"""

cpp_source = r"""
void launch_fused_kernel(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, torch::Tensor& output, int stride, int padding, int K);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fused_op", &launch_fused_kernel); }
"""

fused_ext = load_inline(name='fused_ext', cpp_sources=cpp_source, cuda_sources=cuda_source, 
                        extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, 
                     conv_transpose_groups, conv_transpose_dilation, bias):
    # Output shape calculation
    N, Ci, D, H, W = x.shape
    Co = conv_transpose_weight.shape[1]
    Do = (D - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_output_padding + conv_transpose_weight.shape[2]
    Ho = (H - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_output_padding + conv_transpose_weight.shape[3]
    Wo = (W - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_output_padding + conv_transpose_weight.shape[4]
    
    output = torch.empty((N, Co, Do, Ho, Wo), device=x.device)
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, output, conv_transpose_stride, conv_transpose_padding, conv_transpose_weight.shape[2])
    return output
