# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_043234/code_14.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x', 'add_input']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
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

# --- Custom CUDA Kernel for ConvTranspose3d + Fused Add + HardSwish ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float hardswish_impl(float x) {
    float relu6_val = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * relu6_val * 0.16666667f;
}

// Optimized 3D Transposed Convolution kernel using atomic implementation
__global__ void conv_transpose3d_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    int batch, int in_c, int out_c,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int k, int stride, int pad) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch * in_c * in_d * in_h * in_w;

    for (int i = tid; i < total_threads; i += blockDim.x * gridDim.x) {
        int tmp = i;
        int w_in = tmp % in_w; tmp /= in_w;
        int h_in = tmp % in_h; tmp /= in_h;
        int d_in = tmp % in_d; tmp /= in_d;
        int c_in = tmp % in_c; tmp /= in_c;
        int b = tmp;

        float val = input[i];
        for (int oc = 0; oc < out_c; ++oc) {
            for (int kd = 0; kd < k; ++kd) {
                for (int kh = 0; kh < k; ++kh) {
                    for (int kw = 0; kw < k; ++kw) {
                        int od = d_in * stride + kd - pad;
                        int oh = h_in * stride + kh - pad;
                        int ow = w_in * stride + kw - pad;

                        if (od >= 0 && od < out_d && oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                            int w_idx = oc * (in_c * k * k * k) + c_in * (k * k * k) + kd * (k * k) + kh * k + kw;
                            int out_idx = b * (out_c * out_d * out_h * out_w) + oc * (out_d * out_h * out_w) + od * (out_h * out_w) + oh * out_w + ow;
                            atomicAdd(&output[out_idx], val * weight[w_idx]);
                        }
                    }
                }
            }
        }
    }
}

// Final pass to add bias and apply hardswish
__global__ void post_process_kernel(float* output, const float* bias, const float* add_input, int numel, int out_c, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        int oc = (idx / spatial_size) % out_c;
        float x = output[idx] + bias[oc] + add_input[idx];
        output[idx] = x * hardswish_impl(x);
    }
}

void launch_fused_ops(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, 
                      const at::Tensor& add_input, at::Tensor& output, int stride, int pad) {
    output.fill_(0.0f);
    int batch = input.size(0);
    int in_c = input.size(1);
    int in_d = input.size(2), in_h = input.size(3), in_w = input.size(4);
    int out_c = weight.size(0);
    int out_d = output.size(2), out_h = output.size(3), out_w = output.size(4);
    int k = weight.size(2);

    int threads = 256;
    int blocks = (input.numel() + threads - 1) / threads;
    conv_transpose3d_fused_kernel<<<blocks, threads>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
                                                       add_input.data_ptr<float>(), output.data_ptr<float>(), 
                                                       batch, in_c, out_c, in_d, in_h, in_w, out_d, out_h, out_w, k, stride, pad);
    
    int post_blocks = (output.numel() + threads - 1) / threads;
    post_process_kernel<<<post_blocks, threads>>>(output.data_ptr<float>(), bias.data_ptr<float>(), add_input.data_ptr<float>(), 
                                                  output.numel(), out_c, out_d * out_h * out_w);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_fused_ops(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, 
                      const at::Tensor& add_input, at::Tensor& output, int stride, int pad);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_fused_ops, "Fused ConvTranspose3D + Add + HardSwish");
}
"""

fused_ext = load_inline(name='fused_op_ext', cpp_sources=cpp_source, cuda_sources=cuda_kernel, 
                        extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, add_input, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, conv_transpose_dilation, bias):
    output = torch.zeros((x.size(0), conv_transpose_weight.size(0), x.size(2)*conv_transpose_stride, 
                          x.size(3)*conv_transpose_stride, x.size(4)*conv_transpose_stride), device='cuda')
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, add_input, output, conv_transpose_stride, conv_transpose_padding)
    return output
