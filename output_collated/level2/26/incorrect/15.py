# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_040909/code_15.py
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
# CUDA kernel: Fuses Transposed Conv3D, Add, and HardSwish
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fused_conv_transpose_add_hardswish_kernel(
    const float* __restrict__ input,      // (N, Ci, Di, Hi, Wi)
    const float* __restrict__ weight,     // (Ci, Co, K, K, K) Note: Transposed layout
    const float* __restrict__ add_input,  // (N, Co, Do, Ho, Wo)
    const float* __restrict__ bias,       // (Co)
    float*       __restrict__ output,     // (N, Co, Do, Ho, Wo)
    const int N, const int Ci, const int Co,
    const int Di, const int Hi, const int Wi,
    const int Do, const int Ho, const int Wo,
    const int stride, const int pad, const int ks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Co * Do * Ho * Wo;
    if (idx >= total) return;

    // Decode linear index to (n, co, do, ho, wo)
    int tmp = idx;
    int wo = tmp % Wo; tmp /= Wo;
    int ho = tmp % Ho; tmp /= Ho;
    int do_ = tmp % Do; tmp /= Do;
    int co = tmp % Co; tmp /= Co;
    int n = tmp;

    float val = (bias != nullptr) ? bias[co] : 0.0f;

    // Transposed Conv math:
    // For each output position, we sum contributions from input pixels that map to it.
    // An input pixel at (di, hi, wi) contributes to output at (do, ho, wo) if:
    // do = (di * stride) + kd - pad  => di = (do + pad - kd) / stride
    for (int ic = 0; ic < Ci; ++ic) {
        for (int kd = 0; kd < ks; ++kd) {
            int di_raw = do_ + pad - kd;
            if (di_raw < 0 || di_raw % stride != 0) continue;
            int di = di_raw / stride;
            if (di < 0 || di >= Di) continue;

            for (int kh = 0; kh < ks; ++kh) {
                int hi_raw = ho + pad - kh;
                if (hi_raw < 0 || hi_raw % stride != 0) continue;
                int hi = hi_raw / stride;
                if (hi < 0 || hi >= Hi) continue;

                for (int kw = 0; kw < ks; ++kw) {
                    int wi_raw = wo + pad - kw;
                    if (wi_raw < 0 || wi_raw % stride != 0) continue;
                    int wi = wi_raw / stride;
                    if (wi < 0 || wi >= Wi) continue;

                    // weight layout [Ci, Co, ks, ks, ks]
                    int w_idx = (ic * Co + co) * (ks * ks * ks) + (kd * ks * ks + kh * ks + kw);
                    int i_idx = ((((n * Ci + ic) * Di + di) * Hi + hi) * Wi + wi);
                    val += __ldg(&input[i_idx]) * __ldg(&weight[w_idx]);
                }
            }
        }
    }

    // Add and Activation
    val += __ldg(&add_input[idx]);
    // HardSwish: x * min(max(x + 3, 0), 6) / 6
    float hswish = val * fminf(fmaxf(val + 3.0f, 0.0f), 6.0f) * 0.1666666667f;
    output[idx] = hswish;
}

void launch_fused_kernel(
    const torch::Tensor& input, const torch::Tensor& weight, 
    const torch::Tensor& add_input, const torch::Tensor& bias, 
    torch::Tensor& output, int stride, int padding, int ks) 
{
    int N = input.size(0);
    int Ci = input.size(1);
    int Di = input.size(2), Hi = input.size(3), Wi = input.size(4);
    int Co = weight.size(1);
    int Do = output.size(2), Ho = output.size(3), Wo = output.size(4);
    
    int total = N * Co * Do * Ho * Wo;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    fused_conv_transpose_add_hardswish_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        add_input.data_ptr<float>(), bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), N, Ci, Co, Di, Hi, Wi, Do, Ho, Wo, stride, padding, ks
    );
}
"""

cpp_source = r"""
void launch_fused_kernel(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, torch::Tensor&, int, int, int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_fused_kernel, "Fused TransposedConv3D + Add + HardSwish");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, add_input, *, conv_transpose_weight, conv_transpose_bias, 
                     conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding,
                     conv_transpose_groups, conv_transpose_dilation, bias=None):
    # PyTorch conv_transpose3d weight is [Ci, Co/groups, k, k, k], convert for kernel
    # Input x [N, Ci, Di, Hi, Wi], Weight [Ci, Co, ks, ks, ks]
    weight = conv_transpose_weight.permute(1, 0, 2, 3, 4) # Standardize to [Co, Ci, ...]
    
    D_in, H_in, W_in = x.shape[2:]
    ks = weight.shape[2]
    D_out = (D_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_output_padding + ks
    out = torch.empty((x.shape[0], weight.shape[0], D_out, D_out, D_out), device='cuda')
    
    fused_ext.fused_op(x.contiguous(), weight.contiguous(), add_input.contiguous(), 
                       conv_transpose_bias if conv_transpose_bias is not None else torch.tensor([], device='cuda'),
                       out, conv_transpose_stride, conv_transpose_padding, ks)
    return out
