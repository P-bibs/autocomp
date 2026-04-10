# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_125214/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'min_value', 'divisor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'min_value', 'divisor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

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
    if 'min_value' in flat_state:
        state_kwargs['min_value'] = flat_state['min_value']
    else:
        state_kwargs['min_value'] = getattr(model, 'min_value')
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
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
# 1.  CUDA kernel (device code) – transposed 3‑D convolution in FP16,
#    fused with clamp and division.
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ----------------------------------------------------------------------
// 2.  Transposed 3‑D convolution kernel (FP16)
// ----------------------------------------------------------------------
__global__ void conv_transpose3d_fwd_kernel(
    const half* __restrict__ in,      // input  (N, C_in, D, H, W)
    const half* __restrict__ w,       // weight (C_out, C_in, K, K, K)
    const half* __restrict__ bias,    // bias   (C_out) – may be nullptr
    half* __restrict__ out,           // output (N, C_out, D_out, H_out, W_out)
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int K,                      // kernel size (assumed square)
    const int stride, const int padding, const int output_padding,
    const float min_val, const float divisor)
{
    const int total_out = N * C_out * D_out * H_out * W_out;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_out) return;

    // ------------------------------------------------------------------
    // decode flat output index -> (n, co, d, h, w)
    // ------------------------------------------------------------------
    int rest = tid;
    const int w_out = rest % W_out; rest /= W_out;
    const int h_out = rest % H_out; rest /= H_out;
    const int d_out = rest % D_out; rest /= D_out;
    const int co    = rest % C_out; rest /= C_out;
    const int n     = rest;                     // batch element

    // size of the “upsampled” input grid (without the extra output_padding)
    const int up_d_max = (D_in - 1) * stride + output_padding;
    const int up_h_max = (H_in - 1) * stride + output_padding;
    const int up_w_max = (W_in - 1) * stride + output_padding;

    float acc = 0.0f;

    // ------------------------------------------------------------------
    // 3.  Loop over input channels and kernel positions
    // ------------------------------------------------------------------
    for (int ci = 0; ci < C_in; ++ci) {
        // base offset for weight[co][ci][*][*][*]
        const int w_base = ((co * C_in + ci) * K * K * K);

        // loop over kernel spatial size (K^3)
        for (int kd = 0; kd < K; ++kd) {
            const int up_d = d_out + kd - padding;
            if (up_d < 0 || up_d > up_d_max || (up_d % stride != 0)) continue;
            const int in_d = up_d / stride;

            for (int kh = 0; kh < K; ++kh) {
                const int up_h = h_out + kh - padding;
                if (up_h < 0 || up_h > up_h_max || (up_h % stride != 0)) continue;
                const int in_h = up_h / stride;

                for (int kw = 0; kw < K; ++kw) {
                    const int up_w = w_out + kw - padding;
                    if (up_w < 0 || up_w > up_w_max || (up_w % stride != 0)) continue;
                    const int in_w = up_w / stride;

                    // input element index
                    const int in_idx = (((n * C_in + ci) * D_in + in_d) * H_in + in_h) * W_in + in_w;
                    // weight element index
                    const int w_idx = w_base + ((kd * K + kh) * K + kw);

                    const float in_val = __half2float(in[in_idx]);
                    const float w_val  = __half2float(w[w_idx]);
                    acc += in_val * w_val;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 4.  Add bias (if present)
    // ------------------------------------------------------------------
    if (bias != nullptr) {
        acc += __half2float(bias[co]);
    }

    // ------------------------------------------------------------------
    // 5.  Clamp + scale
    // ------------------------------------------------------------------
    if (acc < min_val) acc = min_val;
    acc /= divisor;

    // ------------------------------------------------------------------
    // 6.  Write result
    // ------------------------------------------------------------------
    const int out_idx = (((n * C_out + co) * D_out + d_out) * H_out + h_out) * W_out + w_out;
    out[out_idx] = __float2half(acc);
}

// ----------------------------------------------------------------------
// 7.  Host‑side launcher that receives torch::Tensor objects
// ----------------------------------------------------------------------
void fused_op(
    int grid, int block,
    at::Tensor in,   // half tensor
    at::Tensor w,    // half tensor
    at::Tensor bias, // half tensor (may be empty)
    at::Tensor out,  // half tensor
    int N, int C_in, int C_out,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int K,
    int stride, int padding, int output_padding,
    float min_val, float divisor)
{
    const half* in_ptr  = (const half*)in.data_ptr();
    const half* w_ptr   = (const half*)w.data_ptr();
    const half* bias_ptr = bias.defined() ? (const half*)bias.data_ptr() : nullptr;
    half* out_ptr = (half*)out.data_ptr();

    conv_transpose3d_fwd_kernel<<<grid, block>>>(
        in_ptr, w_ptr, bias_ptr, out_ptr,
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        K, stride, padding, output_padding,
        min_val, divisor);
}
"""

# ----------------------------------------------------------------------
# 3.  C++ binding (pybind11) – exposes the launcher to Python
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op(
    int grid, int block,
    at::Tensor in,
    at::Tensor w,
    at::Tensor bias,
    at::Tensor out,
    int N, int C_in, int C_out,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int K,
    int stride, int padding, int output_padding,
    float min_val, float divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op,
          "Fused transposed 3‑D convolution (FP16) with clamp and scaling.");
}
"""

# ----------------------------------------------------------------------
# 4.  Compile the extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# ----------------------------------------------------------------------
# 5.  The optimized functional_model that will be imported
# ----------------------------------------------------------------------
def functional_model(
    x,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    min_value,
    divisor,
):
    """
    Fully‑fused transposed 3‑D convolution written in a single custom CUDA
    kernel.  All arithmetic is performed in FP16, the clamp and the division
    by ``divisor`` are performed inside the kernel, and no PyTorch convolution
    primitive is used.
    """
    # ----------------------------------------------------------
    # 5.1  Basic geometry
    # ----------------------------------------------------------
    N, C_in, D_in, H_in, W_in = x.shape
    K = conv_transpose_weight.shape[2]               # kernel size (square)
    C_out = conv_transpose_weight.shape[0]

    # output size (standard PyTorch formula)
    D_out = (D_in - 1) * conv_transpose_stride \
            - 2 * conv_transpose_padding \
            + conv_transpose_dilation * (K - 1) \
            + conv_transpose_output_padding + 1
    H_out = (H_in - 1) * conv_transpose_stride \
            - 2 * conv_transpose_padding \
            + conv_transpose_dilation * (K - 1) \
            + conv_transpose_output_padding + 1
    W_out = (W_in - 1) * conv_transpose_stride \
            - 2 * conv_transpose_padding \
            + conv_transpose_dilation * (K - 1) \
            + conv_transpose_output_padding + 1

    # ----------------------------------------------------------
    # 5.2  Move tensors to the device and convert to FP16
    # ----------------------------------------------------------
    dev = x.device
    x_h     = x.half()
    w_h     = conv_transpose_weight.to(dev).half()
    if conv_transpose_bias is not None:
        b_h = conv_transpose_bias.to(dev).half()
    else:
        b_h = torch.empty(0, dtype=torch.half, device=dev)

    # ----------------------------------------------------------
    # 5.3  Allocate output in FP16
    # ----------------------------------------------------------
    out_h = torch.empty((N, C_out, D_out, H_out, W_out),
                        dtype=torch.half, device=dev)

    # ----------------------------------------------------------
    # 5.4  Launch the fused kernel
    # ----------------------------------------------------------
    block_size = 256
    total_out = N * C_out * D_out * H_out * W_out
    grid = (total_out + block_size - 1) // block_size

    fused_ext.fused_op(
        grid, block_size,
        x_h, w_h, b_h, out_h,
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        K,
        conv_transpose_stride, conv_transpose_padding,
        conv_transpose_output_padding,
        float(min_value), float(divisor)
    )

    # ----------------------------------------------------------
    # 5.5  Return the result as float32 (matches the original API)
    # ----------------------------------------------------------
    return out_h.float()


# ----------------------------------------------------------------------
# 6.  Helper to create the inputs that the evaluator will use
# ----------------------------------------------------------------------
def get_init_inputs():
    # The original test harness expects these numbers (they are not used here)
    return [64, 128, 3, 2, 1, -1.0, 2.0]

def get_inputs():
    batch_size = 16
    in_channels = 64
    depth, height, width = 24, 48, 48
    return [torch.rand(batch_size, in_channels, depth, height, width)]
