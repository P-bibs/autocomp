# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_095834/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a 3D convolution, applies minimum operation along a specific dimension, 
    and then applies softmax.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

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
    # State for conv (nn.Conv3d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
    if 'dim' in flat_state:
        state_kwargs['dim'] = flat_state['dim']
    else:
        state_kwargs['dim'] = getattr(model, 'dim')
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
# CUDA source (kernels + binding code)
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Optimized kernel for 3D Conv (Naive implementation as per instructions)
__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int D_in, const int H_in, const int W_in,
    const int C_out, const int Kd, const int Kh, const int Kw,
    const int stride, const int pad, const int dil,
    const int D_out, const int H_out, const int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * D_out * H_out * W_out;
    if (idx >= total) return;

    int w = idx % W_out; idx /= W_out;
    int h = idx % H_out; idx /= H_out;
    int d = idx % D_out; idx /= D_out;
    int c = idx % C_out; idx /= C_out;
    int n = idx;

    float sum = (bias != nullptr) ? bias[c] : 0.0f;

    for (int ci = 0; ci < C_in; ++ci) {
        for (int kd = 0; kd < Kd; ++kd) {
            int d_in = d * stride + kd * dil - pad;
            if (d_in < 0 || d_in >= D_in) continue;
            for (int kh = 0; kh < Kh; ++kh) {
                int h_in = h * stride + kh * dil - pad;
                if (h_in < 0 || h_in >= H_in) continue;
                for (int kw = 0; kw < Kw; ++kw) {
                    int w_in = w * stride + kw * dil - pad;
                    if (w_in < 0 || w_in >= W_in) continue;

                    int w_idx = ((((c * C_in + ci) * Kd + kd) * Kh + kh) * Kw + kw);
                    int i_idx = ((((n * C_in + ci) * D_in + d_in) * H_in + h_in) * W_in + w_in);
                    sum += weight[w_idx] * input[i_idx];
                }
            }
        }
    }
    output[((((n * C_out + c) * D_out + d) * H_out + h) * W_out + w)] = sum;
}

// Fused kernel: Min over depth (dim 2) followed by Softmax over channels (dim 1)
__global__ void fused_min_softmax_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    const int N, const int C, const int D, const int H, const int W)
{
    int h = blockIdx.y;
    int w = blockIdx.x;
    int n = blockIdx.z;
    
    // Each block calculates the Softmax for a single (n, h, w) stack
    // Shared memory to store min result for C channels
    extern __shared__ float s_data[]; 
    float* min_vals = s_data; 
    
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float m = 1e38f;
        for (int d = 0; d < D; ++d) {
            float val = in[(((n * C + c) * D + d) * H + h) * W + w];
            if (val < m) m = val;
        }
        min_vals[c] = expf(m);
    }
    __syncthreads();

    // Softmax: Needs sum of exps over C
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) sum += min_vals[c];
        for (int c = 0; c < C; ++c) {
            out[((n * C + c) * H + h) * W + w] = min_vals[c] / sum;
        }
    }
}

void launch_conv3d(torch::Tensor in, torch::Tensor w, torch::Tensor b, torch::Tensor out,
                   int N, int C_in, int D_in, int H_in, int W_in, int C_out,
                   int Kd, int Kh, int Kw, int st, int pk, int dl,
                   int D_out, int H_out, int W_out) {
    int total = N * C_out * D_out * H_out * W_out;
    conv3d_kernel<<<(total + 255) / 256, 256>>>(
        in.data_ptr<float>(), w.data_ptr<float>(), b.numel() ? b.data_ptr<float>() : nullptr,
        out.data_ptr<float>(), N, C_in, D_in, H_in, W_in, C_out, Kd, Kh, Kw, st, pk, dl, D_out, H_out, W_out);
}

void launch_fused(torch::Tensor in, torch::Tensor out, int N, int C, int D, int H, int W) {
    dim3 grid(W, H, N);
    fused_min_softmax_kernel<<<grid, 32, C * sizeof(float)>>>(in.data_ptr<float>(), out.data_ptr<float>(), N, C, D, H, W);
}
"""

cpp_source = r"""
void launch_conv3d(torch::Tensor in, torch::Tensor w, torch::Tensor b, torch::Tensor out, int N, int C_in, int D_in, int H_in, int W_in, int C_out, int Kd, int Kh, int Kw, int st, int pk, int dl, int D_out, int H_out, int W_out);
void launch_fused(torch::Tensor in, torch::Tensor out, int N, int C, int D, int H, int W);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d", &launch_conv3d);
    m.def("fused", &launch_fused);
}
"""

fused_ext = load_inline("fused_op", cpp_source, cuda_source, extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups, dim):
    x = x.cuda().contiguous()
    w = conv_weight.cuda().contiguous()
    b = conv_bias.cuda().contiguous() if conv_bias is not None else torch.tensor([], device='cuda')
    
    N, Cin, Din, Hin, Win = x.shape
    Cout, _, Kd, Kh, Kw = w.shape
    Dout = (Din + 2*conv_padding - conv_dilation*(Kd-1) - 1)//conv_stride + 1
    Hout = (Hin + 2*conv_padding - conv_dilation*(Kh-1) - 1)//conv_stride + 1
    Wout = (Win + 2*conv_padding - conv_dilation*(Kw-1) - 1)//conv_stride + 1
    
    conv_out = torch.empty((N, Cout, Dout, Hout, Wout), device='cuda')
    fused_ext.conv3d(x, w, b, conv_out, N, Cin, Din, Hin, Win, Cout, Kd, Kh, Kw, conv_stride, conv_padding, conv_dilation, Dout, Hout, Wout)
    
    final_out = torch.empty((N, Cout, Hout, Wout), device='cuda')
    fused_ext.fused(conv_out, final_out, N, Cout, Dout, Hout, Wout)
    return final_out

def get_init_inputs(): return [3, 24, 3, 2]
def get_inputs(): return [torch.rand(128, 3, 24, 32, 32)]
