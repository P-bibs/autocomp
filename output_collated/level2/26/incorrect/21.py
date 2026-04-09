# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_041736/code_14.py
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

# --- Optimized CUDA Implementation: Tiled Transposed Convolution + Add + H-Swish ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float hardswish(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) * 0.16666667f;
}

// Fused Transposed Conv3D (Naive tiled approach) + Add + HardSwish
__global__ void fused_deconv_add_hswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    int N, int C_in, int C_out, int D, int H, int W,
    int k, int stride, int padding) {

    int out_D = D * stride;
    int out_H = H * stride;
    int out_W = W * stride;
    int out_size = out_D * out_H * out_W;
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N * C_out * out_size) return;

    int tmp = index;
    int w_out = tmp % out_W; tmp /= out_W;
    int h_out = tmp % out_H; tmp /= out_H;
    int d_out = tmp % out_D; tmp /= out_D;
    int c_out = tmp % C_out; tmp /= C_out;
    int n     = tmp;

    float val = bias[c_out];
    
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kd = 0; kd < k; ++kd) {
            for (int kh = 0; kh < k; ++kh) {
                for (int kw = 0; kw < k; ++kw) {
                    int d_in = (d_out + padding - kd);
                    int h_in = (h_out + padding - kh);
                    int w_in = (w_out + padding - kw);
                    
                    if (d_in >= 0 && d_in < D && h_in >= 0 && h_in < H && w_in >= 0 && w_in < W && 
                        d_in % stride == 0 && h_in % stride == 0 && w_in % stride == 0) {
                        
                        int in_idx = (((n * C_in + c_in) * D + d_in/stride) * H + h_in/stride) * W + w_in/stride;
                        int w_idx = (((c_out * C_in + c_in) * k + kd) * k + kh) * k + kw;
                        val += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
    }
    output[index] = hardswish(val + add_input[index]);
}

void launch_fused(const at::Tensor& in, const at::Tensor& w, const at::Tensor& b, 
                  const at::Tensor& add, at::Tensor& out, int stride, int padding) {
    int N = in.size(0); int C_in = in.size(1);
    int C_out = w.size(0); int D = in.size(2); int H = in.size(3); int W = in.size(4);
    int out_size = (D * stride) * (H * stride) * (W * stride);
    int numel = N * C_out * out_size;
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    fused_deconv_add_hswish_kernel<<<blocks, threads>>>(
        in.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
        add.data_ptr<float>(), out.data_ptr<float>(),
        N, C_in, C_out, D, H, W, w.size(2), stride, padding
    );
}
"""

cpp_source = r"""
void launch_fused(const at::Tensor& in, const at::Tensor& w, const at::Tensor& b, 
                  const at::Tensor& add, at::Tensor& out, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_fused, "Fused Deconv3D + Add + HSwish");
}
"""

fused_ext = load_inline('fused', cpp_sources=cpp_source, cuda_sources=cuda_kernel, 
                       extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, add_input, *, conv_transpose_weight, conv_transpose_bias, 
                     conv_transpose_stride, conv_transpose_padding, **kwargs):
    output = torch.empty_like(add_input)
    fused_ext.fused_op(x.contiguous(), conv_transpose_weight.contiguous(), 
                       conv_transpose_bias.contiguous(), add_input.contiguous(), 
                       output, conv_transpose_stride, conv_transpose_padding)
    return output
