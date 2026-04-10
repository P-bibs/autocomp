# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_125214/code_4.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'min_value', 'divisor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'min_value', 'divisor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

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
    if 'min_value' in flat_state:
        state_kwargs['min_value'] = flat_state['min_value']
    else:
        state_kwargs['min_value'] = getattr(model, 'min_value')
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
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

# Define the fused CUDA kernel
# We use a custom kernel for transposed convolution as native conv_transpose3d is excluded per rule 6.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_conv_transpose3d_clamp_divide_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int Ci, int Co,
    int Di, int Hi, int Wi,
    int Do, int Ho, int Wo,
    int k, int s, int p, int op, int g, int d,
    float min_val, float div) {

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * Co * Do * Ho * Wo;
    if (out_idx >= total_elements) return;

    int tmp = out_idx;
    int w_out = tmp % Wo; tmp /= Wo;
    int h_out = tmp % Ho; tmp /= Ho;
    int d_out = tmp % Do; tmp /= Do;
    int c_out = tmp % Co; tmp /= Co;
    int b_idx = tmp;

    int group_idx = c_out / (Co / g);
    int c_in_start = group_idx * (Ci / g);
    int c_in_end = c_in_start + (Ci / g);
    int c_weight = c_out % (Co / g);

    float sum = 0.0f;

    for (int kd = 0; kd < k; kd++) {
        for (int kh = 0; kh < k; kh++) {
            for (int kw = 0; kw < k; kw++) {
                int d_in_raw = d_out + p - kd * d;
                int h_in_raw = h_out + p - kh * d;
                int w_in_raw = w_out + p - kw * d;

                if (d_in_raw >= 0 && d_in_raw % s == 0 &&
                    h_in_raw >= 0 && h_in_raw % s == 0 &&
                    w_in_raw >= 0 && w_in_raw % s == 0) {
                    
                    int d_in = d_in_raw / s;
                    int h_in = h_in_raw / s;
                    int w_in = w_in_raw / s;

                    if (d_in < Di && h_in < Hi && w_in < Wi) {
                        for (int ci = c_in_start; ci < c_in_end; ci++) {
                            int weight_idx = ((c_out * (Ci / g)) + (ci - c_in_start)) * (k * k * k) + (kd * k * k + kh * k + kw);
                            int input_idx = (((b_idx * Ci + ci) * Di + d_in) * Hi + h_in) * Wi + w_in;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }

    sum = fmaxf(sum + bias[c_out], min_val);
    output[out_idx] = sum / div;
}

void launch_fused_op(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
    torch::Tensor& output, int k, int s, int p, int op, int g, int d,
    float min_val, float div) {
    
    int B = input.size(0), Ci = input.size(1), Di = input.size(2), Hi = input.size(3), Wi = input.size(4);
    int Co = weight.size(0);
    int Do = output.size(2), Ho = output.size(3), Wo = output.size(4);

    int total = B * Co * Do * Ho * Wo;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fused_conv_transpose3d_clamp_divide_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        B, Ci, Co, Di, Hi, Wi, Do, Ho, Wo, k, s, p, op, g, d, min_val, div
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_fused_op(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, torch::Tensor& output, int k, int s, int p, int op, int g, int d, float min_val, float div);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fused_op", &launch_fused_op); }
"""

fused_ext = load_inline(name='fused_conv_op', cpp_sources=cpp_source, cuda_sources=cuda_kernel, with_cuda=True, extra_cuda_cflags=['-O3', '--use_fast_math'])

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, conv_transpose_dilation, min_value, divisor):
    B, Ci, Di, Hi, Wi = x.shape
    Co = conv_transpose_weight.size(0)
    k = conv_transpose_weight.size(2)
    Do = (Di - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (k - 1) + conv_transpose_output_padding + 1
    Ho = (Hi - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (k - 1) + conv_transpose_output_padding + 1
    Wo = (Wi - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (k - 1) + conv_transpose_output_padding + 1
    output = torch.empty((B, Co, Do, Ho, Wo), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, output, k, conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, conv_transpose_dilation, min_value, divisor)
    return output
