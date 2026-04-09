# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_030214/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'max_pool1_kernel_size', 'max_pool1_stride', 'max_pool1_padding', 'max_pool1_dilation', 'max_pool1_ceil_mode', 'max_pool1_return_indices', 'max_pool2_kernel_size', 'max_pool2_stride', 'max_pool2_padding', 'max_pool2_dilation', 'max_pool2_ceil_mode', 'max_pool2_return_indices']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)

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
    # State for max_pool1 (nn.MaxPool3d)
    state_kwargs['max_pool1_kernel_size'] = model.max_pool1.kernel_size
    state_kwargs['max_pool1_stride'] = model.max_pool1.stride
    state_kwargs['max_pool1_padding'] = model.max_pool1.padding
    state_kwargs['max_pool1_dilation'] = model.max_pool1.dilation
    state_kwargs['max_pool1_ceil_mode'] = model.max_pool1.ceil_mode
    state_kwargs['max_pool1_return_indices'] = model.max_pool1.return_indices
    # State for max_pool2 (nn.MaxPool3d)
    state_kwargs['max_pool2_kernel_size'] = model.max_pool2.kernel_size
    state_kwargs['max_pool2_stride'] = model.max_pool2.stride
    state_kwargs['max_pool2_padding'] = model.max_pool2.padding
    state_kwargs['max_pool2_dilation'] = model.max_pool2.dilation
    state_kwargs['max_pool2_ceil_mode'] = model.max_pool2.ceil_mode
    state_kwargs['max_pool2_return_indices'] = model.max_pool2.return_indices
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# 1. CUDA source – the fused transposed‑conv + channel‑sum kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose_sum_kernel(
    const float* __restrict__ input,      // (N, C_in, D_in, H_in, W_in)
    const float* __restrict__ weight,     // (C_in, Kd, Kh, Kw) – already summed over out‑channels
    const float* __restrict__ bias,       // (1) – summed bias
    float* __restrict__ output,           // (N, 1, D_out, H_out, W_out)
    const int N, const int C_in,
    const int D_in, const int H_in, const int W_in,
    const int C_out, const int Kd, const int Kh, const int Kw,
    const int stride, const int padding, const int dilation,
    const int D_out, const int H_out, const int W_out)
{
    /* 1‑D grid – one thread per output voxel */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D_out * H_out * W_out) return;

    /* decode (n, d, h, w) */
    int n = idx / (D_out * H_out * W_out);
    int rem = idx % (D_out * H_out * W_out);
    int d = rem / (H_out * W_out);
    rem = rem % (H_out * W_out);
    int h = rem / W_out;
    int w = rem % W_out;

    float sum = 0.0f;

    /* loop over input channels and kernel positions */
    for (int c = 0; c < C_in; ++c) {
        for (int kd = 0; kd < Kd; ++kd) {
            int idx_d = d + padding - kd * dilation;
            if (idx_d % stride != 0) continue;
            int id = idx_d / stride;
            if (id < 0 || id >= D_in) continue;

            for (int kh = 0; kh < Kh; ++kh) {
                int idx_h = h + padding - kh * dilation;
                if (idx_h % stride != 0) continue;
                int ih = idx_h / stride;
                if (ih < 0 || ih >= H_in) continue;

                for (int kw = 0; kw < Kw; ++kw) {
                    int idx_w = w + padding - kw * dilation;
                    if (idx_w % stride != 0) continue;
                    int iw = idx_w / stride;
                    if (iw < 0 || iw >= W_in) continue;

                    /* input index – contiguous (N,C,D,H,W) */
                    int in_idx = ((n * C_in + c) * D_in + id) * H_in * W_in + ih * W_in + iw;
                    /* weight index – flatten (C_in, Kd, Kh, Kw) row‑major */
                    int w_idx = (c * Kd * Kh * Kw + kd * Kh * Kw + kh * Kw + kw);

                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    /* add the (combined) bias */
    sum += bias[0];

    /* store the single‑channel result */
    int out_idx = ((n * C_out + 0) * D_out + d) * H_out * W_out + h * W_out + w;
    output[out_idx] = sum;
}

/* C++ wrapper – called from Python */
void fused_conv_transpose_sum(
    torch::Tensor input,    // (N, C_in, D_in, H_in, W_in)
    torch::Tensor weight,   // (C_in, Kd, Kh, Kw) – summed over out‑channels
    torch::Tensor bias,     // (1)
    torch::Tensor output,   // (N, 1, D_out, H_out, W_out)
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int Kd, int Kh, int Kw,
    int stride, int padding, int dilation,
    int D_out, int H_out, int W_out)
{
    const int threads = 256;
    const int blocks = (N * D_out * H_out * W_out + threads - 1) / threads;
    fused_conv_transpose_sum_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, Kd, Kh, Kw,
        stride, padding, dilation,
        D_out, H_out, W_out);
    cudaDeviceSynchronize();
}
"""

# ----------------------------------------------------------------------
# 2. C++ interface – pybind11 binding
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_conv_transpose_sum(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int Kd, int Kh, int Kw,
    int stride, int padding, int dilation,
    int D_out, int H_out, int W_out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_sum", &fused_conv_transpose_sum,
          "Fused transposed‑convolution + channel‑wise sum");
}
"""

# ----------------------------------------------------------------------
# 3. Build the extension (compile & load)
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_conv',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# 4. The functional model that will be evaluated
# ----------------------------------------------------------------------
def functional_model(
    x,                                 # input tensor (batch, in_channels, D, H, W)
    *,
    conv_transpose_weight,            # (in_channels, out_channels, Kd, Kh, Kw)
    conv_transpose_bias,              # (out_channels,)
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    max_pool1_kernel_size,
    max_pool1_stride,
    max_pool1_padding,
    max_pool1_dilation,
    max_pool1_ceil_mode,
    max_pool1_return_indices,
    max_pool2_kernel_size,
    max_pool2_stride,
    max_pool2_padding,
    max_pool2_dilation,
    max_pool2_ceil_mode,
    max_pool2_return_indices,
):
    # --------------------------------------------------------------
    # 4.1 Collapse the 64 output channels into a single channel
    # --------------------------------------------------------------
    # weight shape: (C_in, C_out, Kd, Kh, Kw)  -> sum over C_out
    combined_weight = conv_transpose_weight.sum(dim=1)      # (C_in, Kd, Kh, Kw)
    combined_bias = conv_transpose_bias.sum(dim=0)          # scalar
    # Make sure bias is a 1‑element tensor for the kernel
    bias_tensor = torch.tensor([combined_bias], dtype=x.dtype, device=x.device)

    # --------------------------------------------------------------
    # 4.2 Compute output size of the transposed convolution
    # --------------------------------------------------------------
    N, C_in, D_in, H_in, W_in = x.shape
    Kd, Kh, Kw = combined_weight.shape[1], combined_weight.shape[2], combined_weight.shape[3]

    stride   = conv_transpose_stride
    padding  = conv_transpose_padding
    dilation = conv_transpose_dilation
    out_pad  = conv_transpose_output_padding

    D_out = (D_in - 1) * stride - 2 * padding + dilation * (Kd - 1) + out_pad + 1
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (Kh - 1) + out_pad + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (Kw - 1) + out_pad + 1

    # --------------------------------------------------------------
    # 4.3 Allocate output tensor and launch the fused kernel
    # --------------------------------------------------------------
    out = torch.zeros((N, 1, D_out, H_out, W_out), dtype=x.dtype, device=x.device)

    fused_ext.fused_conv_transpose_sum(
        x, combined_weight, bias_tensor, out,
        N, C_in, D_in, H_in, W_in,
        1,               # C_out == 1 (we have already summed the channels)
        Kd, Kh, Kw,
        stride, padding, dilation,
        D_out, H_out, W_out)

    x = out

    # --------------------------------------------------------------
    # 4.4 Pooling (unchanged – these are already highly optimised)
    # --------------------------------------------------------------
    x = F.max_pool3d(x,
                     kernel_size=max_pool1_kernel_size,
                     stride=max_pool1_stride,
                     padding=max_pool1_padding,
                     dilation=max_pool1_dilation,
                     ceil_mode=max_pool1_ceil_mode,
                     return_indices=max_pool1_return_indices)

    x = F.max_pool3d(x,
                     kernel_size=max_pool2_kernel_size,
                     stride=max_pool2_stride,
                     padding=max_pool2_padding,
                     dilation=max_pool2_dilation,
                     ceil_mode=max_pool2_ceil_mode,
                     return_indices=max_pool2_return_indices)

    # --------------------------------------------------------------
    # 4.5 Final channel‑wise sum (now a no‑op, kept for semantic equivalence)
    # --------------------------------------------------------------
    x = torch.sum(x, dim=1, keepdim=True)

    return x


# ----------------------------------------------------------------------
# 5. Dummy initialisation helpers (the harness may call them)
# ----------------------------------------------------------------------
batch_size = 16
in_channels = 32
out_channels = 64
depth = height = width = 32
kernel_size = 5
stride = 2
padding = 2


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
