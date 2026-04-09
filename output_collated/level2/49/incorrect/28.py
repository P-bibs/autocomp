# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_094958/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'softmax_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, applies Softmax and Sigmoid.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

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
    # State for softmax (nn.Softmax)
    state_kwargs['softmax_dim'] = model.softmax.dim
    # State for sigmoid (nn.Sigmoid)
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

# ----------------------------------------------------------------------
# CUDA Source: Custom Transposed Convolution Kernel (3x3x3)
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int stride, const int padding, const int dilation) 
{
    // Single thread computes one output value: (n, c_out, d, h, w)
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * C_out * D_out * H_out * W_out;
    if (idx >= total) return;

    int w = idx % W_out;
    int rem = idx / W_out;
    int h = rem % H_out;
    rem /= H_out;
    int d = rem % D_out;
    rem /= D_out;
    int c_out = rem % C_out;
    int n = rem / C_out;

    float val = 0.0f;

    // Kernel size fixed at 3x3x3 for optimized unrolling
    #pragma unroll
    for (int c_in = 0; c_in < C_in; ++c_in) {
        #pragma unroll
        for (int kd = 0; kd < 3; ++kd) {
            int iz_full = d + padding - kd * dilation;
            if (iz_full % stride != 0) continue;
            int iz = iz_full / stride;
            if (iz < 0 || iz >= D_in) continue;

            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                int iy_full = h + padding - kh * dilation;
                if (iy_full % stride != 0) continue;
                int iy = iy_full / stride;
                if (iy < 0 || iy >= H_in) continue;

                #pragma unroll
                for (int kw = 0; kw < 3; ++kw) {
                    int ix_full = w + padding - kw * dilation;
                    if (ix_full % stride != 0) continue;
                    int ix = ix_full / stride;
                    if (ix < 0 || ix >= W_in) continue;

                    int w_idx = (((c_out * C_in + c_in) * 3 + kd) * 3 + kh) * 3 + kw;
                    int i_idx = (((n * C_in + c_in) * D_in + iz) * H_in + iy) * W_in + ix;
                    val += weight[w_idx] * input[i_idx];
                }
            }
        }
    }

    if (bias != nullptr) val += bias[c_out];
    output[idx] = val;
}

void launch_conv_transpose3d(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
    torch::Tensor& output, int stride, int padding, int dilation) 
{
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);
    const int C_out = weight.size(0);
    const int D_out = output.size(2);
    const int H_out = output.size(3);
    const int W_out = output.size(4);

    int64_t total = (int64_t)N * C_out * D_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr, output.data_ptr<float>(),
        N, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out,
        stride, padding, dilation
    );
}
"""

cpp_source = r"""
void launch_conv_transpose3d(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, torch::Tensor& output, int stride, int padding, int dilation);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_conv_transpose3d", &launch_conv_transpose3d, "Custom 3x3x3 ConvTranspose3D");
}
"""

ext = load_inline(name='conv_transpose', cpp_sources=cpp_source, cuda_sources=cuda_source, 
                  extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, softmax_dim):
    # Derived from input tensor and kernel constraints
    kD, kH, kW = 3, 3, 3
    D_out = (x.size(2)-1)*conv_transpose_stride - 2*conv_transpose_padding + conv_transpose_dilation*(kD-1) + conv_transpose_output_padding + 1
    H_out = (x.size(3)-1)*conv_transpose_stride - 2*conv_transpose_padding + conv_transpose_dilation*(kH-1) + conv_transpose_output_padding + 1
    W_out = (x.size(4)-1)*conv_transpose_stride - 2*conv_transpose_padding + conv_transpose_dilation*(kW-1) + conv_transpose_output_padding + 1
    
    out = torch.empty((x.size(0), conv_transpose_weight.size(0), D_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Custom CUDA kernel invocation
    ext.launch_conv_transpose3d(x, conv_transpose_weight, conv_transpose_bias if conv_transpose_bias is not None else torch.empty(0, device=x.device), 
                                out, conv_transpose_stride, conv_transpose_padding, conv_transpose_dilation)
    
    # Standard activations
    x = torch.softmax(out, dim=softmax_dim)
    x = torch.sigmoid(x)
    return x

# Keep global inputs to match environment requirements
batch_size, in_channels, out_channels, D, H, W = 16, 32, 64, 16, 32, 32
def get_init_inputs(): return [in_channels, out_channels, 3, 2, 1, 1]
def get_inputs(): return [torch.rand(batch_size, in_channels, D, H, W).cuda()]
