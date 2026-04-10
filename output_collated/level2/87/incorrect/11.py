# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_141459/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'subtract_value_1', 'subtract_value_2']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'subtract_value_1', 'subtract_value_2']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, subtracts two values, applies Mish activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

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
    # State for conv (nn.Conv2d)
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
    if 'subtract_value_1' in flat_state:
        state_kwargs['subtract_value_1'] = flat_state['subtract_value_1']
    else:
        state_kwargs['subtract_value_1'] = getattr(model, 'subtract_value_1')
    if 'subtract_value_2' in flat_state:
        state_kwargs['subtract_value_2'] = flat_state['subtract_value_2']
    else:
        state_kwargs['subtract_value_2'] = getattr(model, 'subtract_value_2')
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
# CUDA kernel (fused convolution + subtract + mish)
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Fast implementation of mish: x * tanh(softplus(x))
__device__ __forceinline__ float fast_mish(float x) {
    // softplus(x) = log(1 + exp(x)) 
    // For x > 20, softplus(x) is essentially x, and tanh(x) is 1.0. Applying
    // tanh(softplus(x)) directly with log/exp can be numerically unstable.
    // However, x * tanh(softplus(x)) is simply x when x is large.
    float sp = logf(1.0f + expf(fminf(x, 20.0f)));
    return x * tanhf(sp);
}

__global__ void fused_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int H, const int W,
    const int C_out, const int K, const int H_out, const int W_out,
    const int stride, const int padding, const int dilation,
    const float sub1, const float sub2)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    // Unpack linear index to (n, oc, oh, ow)
    const int oc = idx % C_out;
    int tmp = idx / C_out;
    const int ow = tmp % W_out;
    tmp /= W_out;
    const int oh = tmp % H_out;
    const int n = tmp / H_out;

    const int ih_base = oh * stride - padding;
    const int iw_base = ow * stride - padding;

    float sum = (bias != nullptr) ? bias[oc] : 0.0f;

    // Convolution loop
    for (int ic = 0; ic < C_in; ++ic) {
        const float* w_ptr = weight + ((oc * C_in + ic) * K * K);
        const float* in_ptr = input + ((n * C_in + ic) * H * W);

        for (int ki = 0; ki < K; ++ki) {
            int ih = ih_base + ki * dilation;
            if (ih < 0 || ih >= H) continue;
            for (int kj = 0; kj < K; ++kj) {
                int iw = iw_base + kj * dilation;
                if (iw < 0 || iw >= W) continue;
                sum += in_ptr[ih * W + iw] * w_ptr[ki * K + kj];
            }
        }
    }

    sum -= (sub1 + sub2);
    output[idx] = fast_mish(sum);
}

void fused_conv(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int N, int C_in, int H, int W, int C_out, int K, int H_out, int W_out,
    int stride, int padding, int dilation, float sub1, float sub2)
{
    const int total = N * C_out * H_out * W_out;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    
    // Pass raw pointer or nullptr for bias
    float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    
    fused_conv_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr, output.data_ptr<float>(),
        N, C_in, H, W, C_out, K, H_out, W_out, stride, padding, dilation, sub1, sub2
    );
}
"""

cpp_source = r"""
void fused_conv(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int N, int C_in, int H, int W, int C_out, int K, int H_out, int W_out,
    int stride, int padding, int dilation, float sub1, float sub2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &fused_conv, "Fused conv + sub + mish");
}
"""

fused_ext = load_inline(
    name='fused_conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x, *, conv_weight, conv_bias, conv_stride, conv_padding, 
    conv_dilation, conv_groups, subtract_value_1, subtract_value_2,
):
    # This implementation assumes groups=1 as implicit in original context and typical kernel constraints
    N, C_in, H, W = x.shape
    C_out = conv_weight.size(0)
    K = conv_weight.size(2)
    
    H_out = (H + 2 * conv_padding - conv_dilation * (K - 1) - 1) // conv_stride + 1
    W_out = (W + 2 * conv_padding - conv_dilation * (K - 1) - 1) // conv_stride + 1
    
    output = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device='cuda')
    
    fused_ext.fused_conv(
        x.contiguous().cuda(), 
        conv_weight.contiguous().cuda(), 
        conv_bias.cuda() if conv_bias is not None else torch.tensor([], device='cuda'), 
        output,
        N, C_in, H, W, C_out, K, H_out, W_out,
        conv_stride, conv_padding, conv_dilation,
        subtract_value_1, subtract_value_2
    )
    return output
