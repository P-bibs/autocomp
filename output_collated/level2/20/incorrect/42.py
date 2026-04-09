# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_132207/code_19.py
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

# CUDA Kernel implementing 3D Transpose Convolution and Fused Arithmetic
# We use a direct implementation for the transpose convolution and grid-stride tiling
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int C_out, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out, int K, int stride, int padding
) {
    // This represents a custom implementation of ConvTranspose3d
    // For brevity and performance in a single kernel, we perform the accumulation
    // In a production scenario, one would use cuDNN or a GEMM-based approach.
    // Here we use an atomic-based scatter approach (fast for moderate kernel sizes)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * C_in * D_in * H_in * W_in;

    for (int i = tid; i < total_threads; i += blockDim.x * gridDim.x) {
        int tmp = i;
        int w_in = tmp % W_in; tmp /= W_in;
        int h_in = tmp % H_in; tmp /= H_in;
        int d_in = tmp % D_in; tmp /= D_in;
        int c_in = tmp % C_in; tmp /= C_in;
        int b = tmp;

        for (int c_out = 0; c_out < C_out; ++c_out) {
            for (int kd = 0; kd < K; ++kd) {
                for (int kh = 0; kh < K; ++kh) {
                    for (int kw = 0; kw < K; ++kw) {
                        int d_out = d_in * stride - padding + kd;
                        int h_out = h_in * stride - padding + kh;
                        int w_out = w_in * stride - padding + kw;

                        if (d_out >= 0 && d_out < D_out && h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out) {
                            float val = input[i] * weight[c_in * (C_out * K * K * K) + c_out * (K * K * K) + kd * (K * K) + kh * K + kw];
                            atomicAdd(&output[((b * C_out + c_out) * D_out + d_out) * H_out * W_out + h_out * W_out + w_out], val);
                        }
                    }
                }
            }
        }
    }
}

// Fused arithmetic: out = x * (2*x + bias + 1)
__global__ void apply_bias_act_kernel(float* data, const float* bias, int N, int C_out, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        int c = (i / spatial_size) % C_out;
        float x = data[i];
        float b = bias[c];
        data[i] = x * (2.0f * x + b + 1.0f);
    }
}

void launch_fused(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, 
                  int stride, int padding, int K) {
    int B = input.size(0), C_in = input.size(1), D_in = input.size(2), H_in = input.size(3), W_in = input.size(4);
    int C_out = weight.size(1), D_out = output.size(2), H_out = output.size(3), W_out = output.size(4);
    
    // 1. Conv Transpose Accumulation
    dim3 threads(256);
    dim3 blocks(1024);
    fused_conv_transpose_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), // Note: bias handled later
        output.data_ptr<float>(), B, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out, K, stride, padding
    );
    
    // 2. Fused Bias + Arithmetic
    int N = output.numel();
    apply_bias_act_kernel<<<blocks, threads>>>(output.data_ptr<float>(), bias.data_ptr<float>(), N, C_out, D_out*H_out*W_out);
}
"""

cpp_source = """
#include <torch/extension.h>
void launch_fused(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int padding, int K);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch_fused", &launch_fused); }
"""

fused_ext = load_inline(name='fused_ext', cpp_sources=cpp_source, cuda_sources=cuda_kernel, with_cuda=True, extra_cuda_cflags=['-O3', '--use_fast_math'])

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, conv_transpose_padding, 
                     conv_transpose_output_padding, conv_transpose_groups, conv_transpose_dilation, bias):
    out = torch.zeros((x.size(0), conv_transpose_weight.size(1), 
                       (x.size(2)-1)*conv_transpose_stride + conv_transpose_weight.size(2) - 2*conv_transpose_padding,
                       (x.size(3)-1)*conv_transpose_stride + conv_transpose_weight.size(3) - 2*conv_transpose_padding,
                       (x.size(4)-1)*conv_transpose_stride + conv_transpose_weight.size(4) - 2*conv_transpose_padding), device='cuda')
    fused_ext.launch_fused(x, conv_transpose_weight, bias, out, conv_transpose_stride, conv_transpose_padding, conv_transpose_weight.size(2))
    return out
