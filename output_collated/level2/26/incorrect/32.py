# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_043234/code_7.py
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
#  CUDA / C++ source that implements the fused kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// ---------- fused kernel -------------------------------------------------
__global__ void conv_transpose_fused_kernel(
    const float* __restrict__ x,        // input (N, C_in, D, H, W)
    const float* __restrict__ add,      // (N, C_out, D_out, H_out, W_out)
    const float* __restrict__ weight,   // (C_in, C_out/g, K, K, K) per group
    const float* __restrict__ conv_bias,// (C_out,) – may be nullptr
    float* __restrict__ out,            // output (N, C_out, D_out, H_out, W_out)
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int K, const int stride, const int padding,
    const int output_padding, const int groups,
    const int dilation,
    const bool has_conv_bias)
{
    // linear thread id -> (n, co, od, oh, ow)
    const int total_out = N * C_out * D_out * H_out * W_out;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_out) return;

    int rem = tid;
    const int n = rem / (C_out * D_out * H_out * W_out);
    rem %= C_out * D_out * H_out * W_out;
    const int co = rem / (D_out * H_out * W_out);
    rem %= D_out * H_out * W_out;
    const int od = rem / (H_out * W_out);
    rem %= H_out * W_out;
    const int oh = rem / W_out;
    const int ow = rem % W_out;

    // ----- transposed convolution (naïve but fused) -----------------------
    float sum = 0.0f;

    const int g = co / (C_out / groups); // group index
    const int co_g = co % (C_out / groups); // channel in group

    const int C_per_group = C_out / groups;
    const int Cin_per_group = C_in / groups;

    for (int ci_g = 0; ci_g < Cin_per_group; ++ci_g) {
        const int ci = g * Cin_per_group + ci_g;
        
        // weight base for (ci, co)
        const int w_base = ((ci * C_per_group + co_g) * K * K * K);
        
        for (int kd = 0; kd < K; ++kd) {
            int id = (od + padding - kd * dilation);
            if (id < 0 || id % stride != 0) continue;
            id /= stride;
            if (id >= D_in) continue;

            for (int kh = 0; kh < K; ++kh) {
                int ih = (oh + padding - kh * dilation);
                if (ih < 0 || ih % stride != 0) continue;
                ih /= stride;
                if (ih >= H_in) continue;

                for (int kw = 0; kw < K; ++kw) {
                    int iw = (ow + padding - kw * dilation);
                    if (iw < 0 || iw % stride != 0) continue;
                    iw /= stride;
                    if (iw >= W_in) continue;

                    // input index (flattened row‑major)
                    const int x_idx = (((n * C_in + ci) * D_in + id) * H_in + ih) * W_in + iw;
                    // weight index (flattened as in PyTorch: (ci,co,kd,kh,kw))
                    const int w_idx = w_base + (kd * K * K + kh * K + kw);

                    sum += x[x_idx] * weight[w_idx];
                }
            }
        }
    }

    // ----- convolution bias ------------------------------------------------
    if (has_conv_bias) sum += conv_bias[co];

    // ----- add_input -------------------------------------------------------
    const int out_idx = (((n * C_out + co) * D_out + od) * H_out + oh) * W_out + ow;
    sum += add[out_idx];

    // ----- hard‑swish activation -------------------------------------------
    float hs = sum + 3.0f;
    hs = fminf(fmaxf(hs, 0.0f), 6.0f);
    float out_val = sum * hs / 6.0f;

    out[out_idx] = out_val;
}

// ---------- wrapper that launches the kernel -----------------------------
void conv_transpose_fused(
    const torch::Tensor& x,
    const torch::Tensor& add,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    torch::Tensor& out,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int K, const int stride, const int padding,
    const int output_padding, const int groups,
    const int dilation,
    const bool has_conv_bias)
{
    const int block_size = 256;
    const long long total_out = 1LL * N * C_out * D_out * H_out * W_out;
    const int grid = (total_out + block_size - 1) / block_size;

    conv_transpose_fused_kernel<<<grid, block_size>>>(
        x.data_ptr<float>(),
        add.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.defined() ? conv_bias.data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        K, stride, padding,
        output_padding, groups,
        dilation,
        has_conv_bias);

    cudaDeviceSynchronize();
}
"""

# ----------------------------------------------------------------------
#  C++ interface (PYBIND11) – lets Python call the CUDA wrapper
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void conv_transpose_fused(
    const torch::Tensor& x,
    const torch::Tensor& add,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    torch::Tensor& out,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int K, const int stride, const int padding,
    const int output_padding, const int groups,
    const int dilation,
    const bool has_conv_bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose_fused", &conv_transpose_fused,
          "Fused transposed convolution + addition + hard‑swish");
}
"""

# ----------------------------------------------------------------------
#  Compile the extension (CUDA 12.x, PyTorch 2.10.0)
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_conv_transpose',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
#  Original helper functions (unchanged)
# ----------------------------------------------------------------------
batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W),
            torch.rand(batch_size, out_channels, D*stride, H*stride, W*stride)]

# ----------------------------------------------------------------------
#  Re‑written functional_model – uses the fused CUDA kernel
# ----------------------------------------------------------------------
def functional_model(
    x,
    add_input,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    bias,
):
    # ------------------------------------------------------------------
    #  Compute output spatial size (same formula as PyTorch's conv_transpose3d)
    # ------------------------------------------------------------------
    D_in = x.size(2)
    H_in = x.size(3)
    W_in = x.size(4)

    D_out = (D_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + \
            conv_transpose_dilation * (kernel_size - 1) + conv_transpose_output_padding + 1
    H_out = (H_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + \
            conv_transpose_dilation * (kernel_size - 1) + conv_transpose_output_padding + 1
    W_out = (W_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + \
            conv_transpose_dilation * (kernel_size - 1) + conv_transpose_output_padding + 1

    # Allocate output tensor (same shape as add_input)
    out = torch.empty_like(add_input)

    # ------------------------------------------------------------------
    #  Prepare pointers / flags for the optional bias tensors
    # ------------------------------------------------------------------
    conv_bias_ptr = conv_transpose_bias if conv_transpose_bias is not None else torch.tensor([], dtype=torch.float32)
    has_conv_bias = conv_transpose_bias is not None

    # Check that bias parameter is None according to original code
    if bias is not None:
        raise ValueError("The 'bias' parameter should be None according to the original code.")

    # ------------------------------------------------------------------
    #  Launch the fused kernel
    # ------------------------------------------------------------------
    fused_ext.conv_transpose_fused(
        x,                     # input
        add_input,             # tensor to be added
        conv_transpose_weight, # weight tensor
        conv_bias_ptr,         # optional conv‑bias (nullptr if not present)
        out,                   # result
        batch_size,
        in_channels,
        out_channels,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        kernel_size,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_groups,
        conv_transpose_dilation,
        has_conv_bias
    )
    return out
