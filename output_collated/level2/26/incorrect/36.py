# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_043234/code_15.py
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

# ----------------------------------------------------------------------
#  CUDA / C++ source: Fused Transposed Conv3d + Add + HardSwish
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void conv_transpose_fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ add,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ final_bias,
    float* __restrict__ out,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int K, const int stride, const int padding,
    const int dilation,
    const bool has_conv_bias,
    const bool has_final_bias)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_out = N * C_out * D_out * H_out * W_out;
    if (tid >= total_out) return;

    // Decoding linear index to multi-dimensional indices: (n, co, od, oh, ow)
    int n = tid / (C_out * D_out * H_out * W_out);
    int rem = tid % (C_out * D_out * H_out * W_out);
    int co = rem / (D_out * H_out * W_out);
    rem %= (D_out * H_out * W_out);
    int od = rem / (H_out * W_out);
    rem %= (H_out * W_out);
    int oh = rem / W_out;
    int ow = rem % W_out;

    float val = 0.0f;
    for (int ci = 0; ci < C_in; ++ci) {
        for (int kd = 0; kd < K; ++kd) {
            int id_offset = od + padding - kd * dilation;
            if (id_offset < 0 || id_offset % stride != 0) continue;
            int id = id_offset / stride;
            if (id >= D_in) continue;

            for (int kh = 0; kh < K; ++kh) {
                int ih_offset = oh + padding - kh * dilation;
                if (ih_offset < 0 || ih_offset % stride != 0) continue;
                int ih = ih_offset / stride;
                if (ih >= H_in) continue;

                for (int kw = 0; kw < K; ++kw) {
                    int iw_offset = ow + padding - kw * dilation;
                    if (iw_offset < 0 || iw_offset % stride != 0) continue;
                    int iw = iw_offset / stride;
                    if (iw >= W_in) continue;

                    int x_idx = (((n * C_in + ci) * D_in + id) * H_in + ih) * W_in + iw;
                    int w_idx = ((ci * C_out + co) * K * K * K) + (kd * K * K + kh * K + kw);
                    val += x[x_idx] * weight[w_idx];
                }
            }
        }
    }

    if (has_conv_bias) val += conv_bias[co];
    val += add[tid];
    if (has_final_bias) val += final_bias[co];

    // HardSwish: x * relu6(x + 3) / 6
    float hswish = val * fminf(fmaxf(val + 3.0f, 0.0f), 6.0f) * 0.1666666667f;
    out[tid] = hswish;
}

void conv_transpose_fused(
    const torch::Tensor& x, const torch::Tensor& add, const torch::Tensor& weight,
    const torch::Tensor& cb, const torch::Tensor& fb, torch::Tensor& out,
    int N, int Ci, int Co, int Di, int Hi, int Wi, int Do, int Ho, int Wo,
    int K, int s, int p, int d, bool hcb, bool hfb)
{
    const int block_size = 256;
    int64_t total = (int64_t)N * Co * Do * Ho * Wo;
    int grid = (total + block_size - 1) / block_size;
    conv_transpose_fused_kernel<<<grid, block_size>>>(
        x.data_ptr<float>(), add.data_ptr<float>(), weight.data_ptr<float>(),
        hcb ? cb.data_ptr<float>() : nullptr, hfb ? fb.data_ptr<float>() : nullptr,
        out.data_ptr<float>(), N, Ci, Co, Di, Hi, Wi, Do, Ho, Wo, K, s, p, d, hcb, hfb
    );
}
"""

cpp_source = """
void conv_transpose_fused(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                          const torch::Tensor&, const torch::Tensor&, torch::Tensor&,
                          int, int, int, int, int, int, int, int, int, int, int, int, int, bool, bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &conv_transpose_fused, "Fused op");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, add_input, *, conv_transpose_weight, conv_transpose_bias, 
                     conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding, 
                     conv_transpose_groups, conv_transpose_dilation, bias):
    N, Ci, Di, Hi, Wi = x.shape
    Co = conv_transpose_weight.shape[1]
    K = conv_transpose_weight.shape[2]
    
    Do = (Di - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1
    Ho = (Hi - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1
    Wo = (Wi - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1
    
    out = torch.empty((N, Co, Do, Ho, Wo), device=x.device, dtype=x.dtype)
    cb = conv_transpose_bias if conv_transpose_bias is not None else torch.tensor([], device=x.device)
    fb = bias if bias is not None else torch.tensor([], device=x.device)
    
    fused_ext.fused_op(x, add_input, conv_transpose_weight, cb, fb, out,
                       N, Ci, Co, Di, Hi, Wi, Do, Ho, Wo, K, 
                       conv_transpose_stride, conv_transpose_padding, conv_transpose_dilation,
                       conv_transpose_bias is not None, bias is not None)
    return out
