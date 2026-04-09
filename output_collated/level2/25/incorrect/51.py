# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_090933/code_6.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

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

# -------------------------------------------------------------------------
# CUDA source: fused convolution + channel‑min + double‑tanh kernel
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// ---------------------------------------------------------------------
// Fused kernel: 2‑D convolution -> min over output channels -> tanh -> tanh
// ---------------------------------------------------------------------
__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const int N, const int C_in, const int C_out,
    const int H, const int W, const int K,
    const int stride, const int padding) {

    const int OH = (H + 2*padding - K) / stride + 1;
    const int OW = (W + 2*padding - K) / stride + 1;

    const int total = N * OH * OW;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    // -----------------------------------------------------------------
    // Decode linear index into (n, oh, ow)
    // -----------------------------------------------------------------
    int n = tid / (OH * OW);
    int rest = tid % (OH * OW);
    int oh = rest / OW;
    int ow = rest % OW;

    // -----------------------------------------------------------------
    // Compute the minimum across all output channels for this pixel
    // -----------------------------------------------------------------
    float min_val = 1e38f;               // +infinity

    for (int co = 0; co < C_out; ++co) {
        float sum = bias[co];            // add bias up‑front

        // loop over input channels and kernel spatial positions
        for (int ci = 0; ci < C_in; ++ci) {
            for (int kh = 0; kh < K; ++kh) {
                int ih = oh * stride + kh - padding;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < K; ++kw) {
                    int iw = ow * stride + kw - padding;
                    if (iw < 0 || iw >= W) continue;

                    // read input (cached by texture/L1)
                    float in_val = __ldg(&x[((n * C_in + ci) * H + ih) * W + iw]);
                    // read weight (also cached)
                    float w_val = __ldg(&weight[(((co * C_in + ci) * K + kh) * K + kw)]);
                    sum += in_val * w_val;
                }
            }
        }
        // keep the smallest result among all output channels
        min_val = fminf(min_val, sum);
    }

    // -----------------------------------------------------------------
    // Double tanh activation (fast math)
    // -----------------------------------------------------------------
    min_val = tanhf(min_val);
    min_val = tanhf(min_val);

    // Write final scalar result
    out[tid] = min_val;
}

// ---------------------------------------------------------------------
// Host wrapper – called from Python
// ---------------------------------------------------------------------
void fused_conv_min_tanh(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out,
    int stride,
    int padding) {

    const int N   = x.size(0);
    const int C_in = x.size(1);
    const int H   = x.size(2);
    const int W   = x.size(3);
    const int C_out = weight.size(0);
    const int K   = weight.size(2);

    const int OH = (H + 2*padding - K) / stride + 1;
    const int OW = (W + 2*padding - K) / stride + 1;

    const int total = N * OH * OW;
    const int block_size = 256;                 // multiple of 32 warps
    const int grid = (total + block_size - 1) / block_size;

    fused_conv_min_tanh_kernel<<<grid, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C_in, C_out, H, W, K, stride, padding);

    cudaDeviceSynchronize();
}
"""

# -------------------------------------------------------------------------
# C++ interface (PyBind11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_min_tanh(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out,
    int stride,
    int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_min_tanh", &fused_conv_min_tanh,
          "Fused convolution + channel‑wise min + double tanh");
}
"""

# -------------------------------------------------------------------------
# Compile the inline CUDA extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# functional_model – the only function that will be imported
# -------------------------------------------------------------------------
def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,   # not used – assumed = 1
    conv_groups,     # not used – assumed = 1
):
    """
    Fused implementation:
        • 2‑D convolution (stride, padding)
        • min over output channels
        • double tanh
    All three steps run in a single CUDA kernel.
    """
    # Compute output spatial size (same formula as in the kernel)
    K = conv_weight.size(2)                # kernel height/width (square)
    OH = (x.size(2) + 2 * conv_padding - K) // conv_stride + 1
    OW = (x.size(3) + 2 * conv_padding - K) // conv_stride + 1

    # Allocate output: (N, 1, OH, OW)
    out = torch.empty((x.size(0), 1, OH, OW), device=x.device, dtype=x.dtype)

    # Launch the fused kernel
    fused_ext.fused_conv_min_tanh(
        x, conv_weight, conv_bias, out,
        conv_stride, conv_padding)

    return out


# -------------------------------------------------------------------------
# Test‑harness (not required for grading, but useful for local validation)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    batch_size = 128
    in_channels = 16
    out_channels = 64
    height = width = 256
    kernel_size = 3

    torch.manual_seed(0)
    x = torch.rand(batch_size, in_channels, height, width, device='cuda')
    weight = torch.rand(out_channels, in_channels, kernel_size, kernel_size,
                        device='cuda')
    bias = torch.rand(out_channels, device='cuda')

    out = functional_model(
        x,
        conv_weight=weight,
        conv_bias=bias,
        conv_stride=1,
        conv_padding=1,
        conv_dilation=1,
        conv_groups=1,
    )
    print("Output shape:", out.shape)   # (128, 1, 256, 256)
