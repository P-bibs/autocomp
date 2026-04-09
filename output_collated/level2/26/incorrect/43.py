# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_044115/code_14.py
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

# --- CUDA Kernel ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float hardswish_impl(float x) {
    float relu6_val = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * relu6_val * 0.16666667f;
}

__global__ void fused_conv_tr_hs_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int k, const int s, const int p) {

    extern __shared__ float s_weight[];
    
    // Cooperative Weight Loading
    int w_size = C_in * C_out * k * k * k;
    for (int i = threadIdx.x; i < w_size; i += blockDim.x) s_weight[i] = weight[i];
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out * D_out * H_out * W_out) return;

    int tmp = idx;
    int w_out = tmp % W_out; tmp /= W_out;
    int h_out = tmp % H_out; tmp /= H_out;
    int d_out = tmp % D_out; tmp /= D_out;
    int co = tmp % C_out; tmp /= C_out;
    int n = tmp;

    float sum = 0.0f;
    for (int ci = 0; ci < C_in; ++ci) {
        for (int kd = 0; kd < k; ++kd) {
            int d_in = d_out + p - kd;
            if (d_in % s != 0) continue;
            d_in /= s;
            if (d_in < 0 || d_in >= D_in) continue;

            for (int kh = 0; kh < k; ++kh) {
                int h_in = h_out + p - kh;
                if (h_in % s != 0) continue;
                h_in /= s;
                if (h_in < 0 || h_in >= H_in) continue;

                for (int kw = 0; kw < k; ++kw) {
                    int w_in = w_out + p - kw;
                    if (w_in % s != 0) continue;
                    w_in /= s;
                    if (w_in < 0 || w_in >= W_in) continue;

                    int in_idx = (((n * C_in + ci) * D_in + d_in) * H_in + h_in) * W_in + w_in;
                    int w_idx = ((ci * C_out + co) * k * k * k) + (kd * k * k + kh * k + kw);
                    sum += input[in_idx] * s_weight[w_idx];
                }
            }
        }
    }
    output[idx] = hardswish_impl(sum + add_input[idx]);
}

void launch_fused_conv_tr_hs(
    const at::Tensor& in, const at::Tensor& wt, const at::Tensor& add, at::Tensor& out,
    int N, int Cin, int Cout, int Din, int Hin, int Win, int Dout, int Hout, int Wout,
    int k, int s, int p) {
    const int numel = N * Cout * Dout * Hout * Wout;
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    const int smem = Cin * Cout * k * k * k * sizeof(float);
    
    fused_conv_tr_hs_kernel<<<blocks, threads, smem>>>(
        in.data_ptr<float>(), wt.data_ptr<float>(), add.data_ptr<float>(), out.data_ptr<float>(),
        N, Cin, Cout, Din, Hin, Win, Dout, Hout, Wout, k, s, p
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_fused_conv_tr_hs(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int, int, int, int, int, int, int, int, int, int, int, int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fused", &launch_fused_conv_tr_hs); }
"""

fused_ext = load_inline(name='fused_conv_tr_hs', cpp_sources=cpp_source, cuda_sources=cuda_kernel, 
                       extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, add_input, *, conv_transpose_weight, conv_transpose_bias, 
                     conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding, 
                     conv_transpose_groups, conv_transpose_dilation, bias):
    N, C_in, D_in, H_in, W_in = x.shape
    C_out, _, k, _, _ = conv_transpose_weight.shape
    s, p, op, d = conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding, conv_transpose_dilation
    D_out = (D_in - 1) * s - 2 * p + d * (k - 1) + op + 1
    H_out = (H_in - 1) * s - 2 * p + d * (k - 1) + op + 1
    W_out = (W_in - 1) * s - 2 * p + d * (k - 1) + op + 1
    out = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device)
    fused_ext.fused(x.contiguous(), conv_transpose_weight.contiguous(), add_input.contiguous(), out,
                    N, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out, k, s, p)
    return out
