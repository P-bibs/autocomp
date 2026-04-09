# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_083856/code_6.py
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
# 1.  CUDA source – the fused kernel that implements the convolution,
#     the min‑reduction over output channels and the double tanh.
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused kernel:  conv → min over C_out → tanh → tanh
__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x,       // input  (N, C_in, H, W)
    const float* __restrict__ weight,  // weight (C_out, C_in, K, K) flattened
    const float* __restrict__ bias,    // bias   (C_out)
    float*       __restrict__ output,  // output (N, 1, OH, OW)
    const int N, const int C_in, const int C_out,
    const int H, const int W, const int K,
    const int stride, const int padding,
    const int OH, const int OW)
{
    // each thread handles one spatial output position (n, oh, ow)
    const int total = N * OH * OW;
    const int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const int n  = tid / (OH * OW);
    const int rem = tid % (OH * OW);
    const int oh = rem / OW;
    const int ow = rem % OW;

    // -----------------------------------------------------------------
    // 1) im2col – build the patch vector p[ C_in * K * K ]  (max 144)
    // -----------------------------------------------------------------
    float p[144];                       // compile‑time max = 16*3*3
    const int Kk = C_in * K * K;        // actual length of the patch

    #pragma unroll
    for (int ci = 0; ci < C_in; ++ci) {
        #pragma unroll
        for (int kh = 0; kh < K; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < K; ++kw) {
                const int ih = oh * stride + kh - padding;
                const int iw = ow * stride + kw - padding;
                float v = 0.0f;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    // read‑only cache load
                    v = __ldg(&x[((n * C_in + ci) * H + ih) * W + iw]);
                }
                p[ci * K * K + kh * K + kw] = v;
            }
        }
    }

    // -----------------------------------------------------------------
    // 2) compute convolution for every output channel and keep the min
    // -----------------------------------------------------------------
    float min_val = 1e38f;   // large positive

    #pragma unroll
    for (int co = 0; co < C_out; ++co) {
        float sum = __ldg(&bias[co]);
        const int w_off = co * Kk;
        #pragma unroll
        for (int k = 0; k < Kk; ++k) {
            // weight is also read‑only → use __ldg
            sum += p[k] * __ldg(&weight[w_off + k]);
        }
        if (sum < min_val) min_val = sum;
    }

    // -----------------------------------------------------------------
    // 3) double tanh activation
    // -----------------------------------------------------------------
    float out_val = tanhf(tanhf(min_val));

    // -----------------------------------------------------------------
    // 4) store result – shape (N,1,OH,OW)
    // -----------------------------------------------------------------
    output[(n * OH + oh) * OW + ow] = out_val;
}

// Host function that launches the kernel
void fused_op_forward(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    const int N, const int C_in, const int C_out,
    const int H, const int W, const int K,
    const int stride, const int padding,
    const int OH, const int OW)
{
    const int block_dim = 256;
    const int grid_dim  = (N * OH * OW + block_dim - 1) / block_dim;

    fused_conv_min_tanh_kernel<<<grid_dim, block_dim>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out, H, W, K, stride, padding, OH, OW);
}
"""

# -------------------------------------------------------------------------
# 2.  C++ binding – exposes the host function to Python
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    const int N, const int C_in, const int C_out,
    const int H, const int W, const int K,
    const int stride, const int padding,
    const int OH, const int OW);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward,
          "Fused convolution + channel‑min + double tanh forward");
}
"""

# -------------------------------------------------------------------------
# 3.  Compile the inline CUDA extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name="fused_op",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
)

# -------------------------------------------------------------------------
# 4.  Functional model – the only entry point that will be imported
# -------------------------------------------------------------------------
def functional_model(
    x: torch.Tensor,
    *,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
):
    """
    Fused implementation:
        1) 2‑D convolution (stride, padding)
        2) min over output channels
        3) tanh(tanh(min))

    The kernel does everything in a single launch, using the __ldg intrinsic
    for read‑only data and avoiding any intermediate global‑memory traffic.
    """
    # --------------------------------------------------------------
    # Extract geometry
    # --------------------------------------------------------------
    N, C_in, H, W = x.shape
    C_out = conv_weight.shape[0]
    K = conv_weight.shape[2]          # square kernel assumed

    # stride / padding may be int or tuple – normalise to int
    stride = conv_stride[0] if isinstance(conv_stride, (list, tuple)) else conv_stride
    padding = conv_padding[0] if isinstance(conv_padding, (list, tuple)) else conv_padding

    # -----------------------------------------------------------------
    # Basic sanity checks – the kernel does not support dilation / groups
    # -----------------------------------------------------------------
    if conv_dilation != (1, 1) and conv_dilation != 1:
        raise NotImplementedError("Dilation != 1 is not supported by the custom kernel")
    if conv_groups != 1:
        raise NotImplementedError("Groups != 1 is not supported by the custom kernel")

    # Output spatial size (same formula as PyTorch)
    OH = (H + 2 * padding - K) // stride + 1
    OW = (W + 2 * padding - K) // stride + 1

    # Allocate output tensor with shape (N,1,OH,OW)
    output = torch.empty((N, 1, OH, OW), dtype=x.dtype, device=x.device)

    # --------------------------------------------------------------
    # Launch the fused CUDA kernel
    # --------------------------------------------------------------
    fused_ext.fused_op(
        x,                      # input  (N, C_in, H, W)
        conv_weight,            # weight (C_out, C_in, K, K)
        conv_bias,              # bias   (C_out)
        output,                 # output (N,1,OH,OW)
        N, C_in, C_out,
        H, W, K,
        stride, padding,
        OH, OW,
    )
    return output
