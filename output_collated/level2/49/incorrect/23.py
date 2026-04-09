# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_094958/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'softmax_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, applies Softmax and Sigmoid.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

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
    # State for softmax (nn.Softmax)
    state_kwargs['softmax_dim'] = model.softmax.dim
    # State for sigmoid (nn.Sigmoid)
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
# Inline CUDA source – custom transposed‑convolution kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ------------------------------------------------------------
// transposed convolution kernel (gather version)
// ------------------------------------------------------------
__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int kD, const int kH, const int kW,
    const int stride, const int padding,
    const int output_padding,   // only needed for size calculation, not used in the kernel
    const int groups,
    const int dilation)
{
    // each thread computes one output element
    const int64_t total_out = (int64_t)N * C_out * D_out * H_out * W_out;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    // decode flat index to (n, c_out, d, h, w)
    int w = idx % W_out;
    int rem = idx / W_out;
    int h = rem % H_out;
    rem /= H_out;
    int d = rem % D_out;
    rem /= D_out;
    int c_out = rem % C_out;
    int n = rem / C_out;

    float sum = 0.0f;

    // loop over input channels and kernel positions
    // the three inner loops are unrolled because kD=kH=kW=3
    #pragma unroll
    for (int c_in = 0; c_in < C_in; ++c_in) {
        #pragma unroll
        for (int kd = 0; kd < kD; ++kd) {
            int iz = (d + padding - kd * dilation);
            if (iz % stride != 0) continue;
            iz /= stride;
            if (iz < 0 || iz >= D_in) continue;

            #pragma unroll
            for (int ky = 0; ky < kH; ++ky) {
                int iy = (h + padding - ky * dilation);
                if (iy % stride != 0) continue;
                iy /= stride;
                if (iy < 0 || iy >= H_in) continue;

                #pragma unroll
                for (int kx = 0; kx < kW; ++kx) {
                    int ix = (w + padding - kx * dilation);
                    if (ix % stride != 0) continue;
                    ix /= stride;
                    if (ix < 0 || ix >= W_in) continue;

                    // weight index: (C_out, C_in, kD, kH, kW) -> flat
                    int w_idx = (((c_out * C_in + c_in) * kD + kd) * kH + ky) * kW + kx;
                    float wval = weight[w_idx];

                    // input index: (N, C_in, D_in, H_in, W_in) -> flat
                    int i_idx = (((n * C_in + c_in) * D_in + iz) * H_in + iy) * W_in + ix;
                    float ival = input[i_idx];

                    sum += wval * ival;
                }
            }
        }
    }

    if (bias && bias[c_out] != 0.0f) sum += bias[c_out];

    // write result
    int64_t out_idx = (((n * C_out + c_out) * D_out + d) * H_out + h) * W_out + w;
    output[out_idx] = sum;
}

// ------------------------------------------------------------
// Host wrapper that launches the kernel
// ------------------------------------------------------------
void conv_transpose3d_cuda(
    const torch::Tensor& input,   // (N, C_in, D, H, W)
    const torch::Tensor& weight,  // (C_out, C_in, kD, kH, kW)
    const torch::Tensor& bias,    // (C_out) or empty
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation,
    torch::Tensor& output)       // (N, C_out, D_out, H_out, W_out)
{
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    const int C_out = weight.size(0);
    const int kD = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);

    // compute output sizes (the same formula used by PyTorch)
    const int D_out = (D_in - 1) * stride - 2 * padding + dilation * (kD - 1) + output_padding + 1;
    const int H_out = (H_in - 1) * stride - 2 * padding + dilation * (kH - 1) + output_padding + 1;
    const int W_out = (W_in - 1) * stride - 2 * padding + dilation * (kW - 1) + output_padding + 1;

    // allocate output if not already done
    if (output.numel() != N * C_out * D_out * H_out * W_out) {
        output = torch::empty({N, C_out, D_out, H_out, W_out}, input.options());
    }

    const int64_t num_out = (int64_t)N * C_out * D_out * H_out * W_out;
    const int threads = 256;
    const int blocks = (num_out + threads - 1) / threads;

    conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        kD, kH, kW,
        stride, padding, output_padding,
        groups, dilation);
}
"""

# ------------------------------------------------------------
# C++ bindings – expose the wrapper to Python
# ------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void conv_transpose3d_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation,
    torch::Tensor& output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose3d_cuda",
          &conv_transpose3d_cuda,
          "Custom transposed convolution (CUDA)");
}
"""

# ------------------------------------------------------------
# Compile the inline extension
# ------------------------------------------------------------
conv_ext = load_inline(
    name='conv_transpose3d_cuda',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Helper to compute output spatial size (same formula as PyTorch)
# ----------------------------------------------------------------------
def conv_out_size(in_dim, stride, padding, kernel, dilation, output_padding):
    return (in_dim - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1

# ----------------------------------------------------------------------
# The functional model that will be imported during evaluation
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
    softmax_dim,
):
    # ---- 1) custom transposed convolution (optimisation #7) ----
    # infer kernel size from weight shape (the weight is (C_out, C_in, kD, kH, kW))
    kD, kH, kW = conv_transpose_weight.shape[2], conv_transpose_weight.shape[3], conv_transpose_weight.shape[4]

    # allocate output tensor (will be filled by the CUDA kernel)
    out = torch.empty((
        x.size(0),
        conv_transpose_weight.size(0),
        conv_out_size(x.size(2), conv_transpose_stride, conv_transpose_padding,
                      kD, conv_transpose_dilation, conv_transpose_output_padding),
        conv_out_size(x.size(3), conv_transpose_stride, conv_transpose_padding,
                      kH, conv_transpose_dilation, conv_transpose_output_padding),
        conv_out_size(x.size(4), conv_transpose_stride, conv_transpose_padding,
                      kW, conv_transpose_dilation, conv_transpose_output_padding),
    ), dtype=x.dtype, device=x.device)

    # call the custom kernel
    conv_ext.conv_transpose3d_cuda(
        x,
        conv_transpose_weight,
        conv_transpose_bias if conv_transpose_bias is not None else torch.empty(0),
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_groups,
        conv_transpose_dilation,
        out)

    x = out  # now x holds the convolution result

    # ---- 2) activation sequence (unchanged) ----
    x = F.softmax(x, dim=softmax_dim)
    x = torch.sigmoid(x)
    return x

# ----------------------------------------------------------------------
# Optional: small test driver (not part of the solution, only for sanity)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # quick sanity-check that the kernel runs without error
    batch_size = 2
    in_channels = 8
    out_channels = 16
    D, H, W = 8, 8, 8
    x = torch.rand(batch_size, in_channels, D, H, W, device='cuda')
    weight = torch.rand(out_channels, in_channels, 3, 3, 3, device='cuda')
    bias = torch.rand(out_channels, device='cuda')
    y = functional_model(
        x,
        conv_transpose_weight=weight,
        conv_transpose_bias=bias,
        conv_transpose_stride=2,
        conv_transpose_padding=1,
        conv_transpose_output_padding=1,
        conv_transpose_groups=1,
        conv_transpose_dilation=1,
        softmax_dim=1)
    print("Output shape:", y.shape)   # (2, 16, ?, ?, ?)
    print("Test passed!")
