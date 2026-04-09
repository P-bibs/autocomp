# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_112403/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'bias_shape', 'stride', 'padding', 'output_padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
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
    # State for conv_transpose (nn.ConvTranspose2d)
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
# CUDA kernel: Fused Transposed Conv2d + Bias + Tanh
# Optimized for performance on RTX 2080 Ti
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_transpose_conv_bias_tanh_kernel(
    float* __restrict__ out,
    const float* __restrict__ inp,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ final_bias,
    const int N, const int C_in, const int C_out,
    const int H, const int W, const int K,
    const int stride, const int padding,
    const int groups,
    const int out_h, const int out_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * out_h * out_w;
    if (idx >= total_elements) return;

    // Unpack indices: N, C_out, OH, OW
    int tmp = idx;
    int ow = tmp % out_w; tmp /= out_w;
    int oh = tmp % out_h; tmp /= out_h;
    int oc = tmp % C_out; tmp /= C_out;
    int n = tmp;

    // Accumulate convolution
    float sum = (conv_bias != nullptr) ? conv_bias[oc] : 0.0f;

    const int ic_per_group = C_in / groups;
    const int start_ic = (oc / (C_out / groups)) * ic_per_group;

    for (int ic_rel = 0; ic_rel < ic_per_group; ++ic_rel) {
        int ic = start_ic + ic_rel;
        
        // Weight pointer: [C_in, C_out/groups, K, K] layout for Transposed Conv
        // Usually weights are [C_in, C_out/groups, K, K]
        const float* w_ptr = weight + (ic * (C_out / groups) + (oc % (C_out / groups))) * K * K;

        for (int kh = 0; kh < K; ++kh) {
            int hi = (oh + padding - kh);
            if (hi % stride != 0) continue;
            hi /= stride;
            if (hi < 0 || hi >= H) continue;

            for (int kw = 0; kw < K; ++kw) {
                int wi = (ow + padding - kw);
                if (wi % stride != 0) continue;
                wi /= stride;
                if (wi < 0 || wi >= W) continue;

                float inp_val = inp[((n * C_in + ic) * H + hi) * W + wi];
                sum += inp_val * w_ptr[kh * K + kw];
            }
        }
    }

    if (final_bias != nullptr) sum -= final_bias[oc];
    out[idx] = tanhf(sum);
}

void fused_op_forward(
    torch::Tensor out, torch::Tensor inp, torch::Tensor weight,
    torch::Tensor conv_bias, torch::Tensor final_bias,
    int N, int C_in, int C_out, int H, int W, int K,
    int stride, int padding, int groups, int out_h, int out_w)
{
    int total = N * C_out * out_h * out_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    fused_transpose_conv_bias_tanh_kernel<<<blocks, threads>>>(
        out.data_ptr<float>(), inp.data_ptr<float>(), weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(), final_bias.data_ptr<float>(),
        N, C_in, C_out, H, W, K, stride, padding, groups, out_h, out_w
    );
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor out, torch::Tensor inp, torch::Tensor weight,
                      torch::Tensor conv_bias, torch::Tensor final_bias,
                      int N, int C_in, int C_out, int H, int W, int K,
                      int stride, int padding, int groups, int out_h, int out_w);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Transposed Conv");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias):
    device = torch.device('cuda')
    x = x.to(device)
    
    # Pre-process parameters
    N, C_in, H, W = x.shape
    C_out = conv_transpose_weight.shape[1] * conv_transpose_groups
    K = conv_transpose_weight.shape[2]
    
    out_h = (H - 1) * conv_transpose_stride - 2 * conv_transpose_padding + K + conv_transpose_output_padding
    out_w = (W - 1) * conv_transpose_stride - 2 * conv_transpose_padding + K + conv_transpose_output_padding
    
    out = torch.empty((N, C_out, out_h, out_w), device=device)
    
    fused_ext.fused_op(
        out, x, conv_transpose_weight, conv_transpose_bias, bias,
        N, C_in, C_out, H, W, K, 
        conv_transpose_stride, conv_transpose_padding, conv_transpose_groups, out_h, out_w
    )
    return out
