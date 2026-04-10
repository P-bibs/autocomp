# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_134740/code_13.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, followed by a sum, 
    a residual add, a multiplication, and another residual add.
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

# -------------------------------------------------------------------------
# Fused CUDA kernel – transposed 3D convolution + element-wise fused op
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Grid-stride kernel: one thread per output element, fully fuses the
// transposed convolution with the post-processing formula
//   ((x + bias) + x) * x + x   ==   2*x*x + bias*x + x
__global__ void fused_transpose_conv_kernel(
    const float* __restrict__ input,      // (N, C_in, D_in, H_in, W_in)
    const float* __restrict__ weight,     // (C_in, C_out, Kd, Kh, Kw)
    const float* __restrict__ conv_bias,  // (C_out) – bias of the transposed conv (may be zero)
    const float* __restrict__ bias2,      // (C_out) – bias used in the fused element-wise op
    float* __restrict__ output,           // (N, C_out, D_out, H_out, W_out)
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int stride, const int padding,
    const int /*output_padding*/, const int dilation,
    const int Kd, const int Kh, const int Kw)
{
    const int total = N * C_out * D_out * H_out * W_out;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride_grid = blockDim.x * gridDim.x;

    // -----------------------------------------------------------------
    // Grid-stride loop – each thread processes multiple output elements
    // -----------------------------------------------------------------
    for (int idx = tid; idx < total; idx += stride_grid) {

        // ----- decode linear index to (n, c, d, h, w) -----
        int rem = idx;
        int n = rem / (C_out * D_out * H_out * W_out);
        rem   = rem % (C_out * D_out * H_out * W_out);
        int c = rem / (D_out * H_out * W_out);
        rem   = rem % (D_out * H_out * W_out);
        int d = rem / (H_out * W_out);
        rem   = rem % (H_out * W_out);
        int h = rem / W_out;
        int w = rem % W_out;

        // ----- decide which kernel offsets are active (stride==2, padding==1) -----
        int parity_d = (d + padding) & 1;
        int parity_h = (h + padding) & 1;
        int parity_w = (w + padding) & 1;

        int kd_arr[2], kh_arr[2], kw_arr[2];
        int num_kd, num_kh, num_kw;

        if (parity_d == 0) { kd_arr[0] = 1; num_kd = 1; }
        else               { kd_arr[0] = 0; kd_arr[1] = 2; num_kd = 2; }

        if (parity_h == 0) { kh_arr[0] = 1; num_kh = 1; }
        else               { kh_arr[0] = 0; kh_arr[1] = 2; num_kh = 2; }

        if (parity_w == 0) { kw_arr[0] = 1; num_kw = 1; }
        else               { kw_arr[0] = 0; kw_arr[1] = 2; num_kw = 2; }

        // ----- accumulate the transposed convolution -----
        float conv_sum = 0.0f;
        const size_t batch_base = (size_t)n * C_in * D_in * H_in * W_in;

        for (int ic = 0; ic < C_in; ++ic) {
            const float* in_ptr = input + batch_base + (size_t)ic * D_in * H_in * W_in;
            const float* w_ptr  = weight + ((size_t)ic * C_out + c) * (size_t)Kd * Kh * Kw;

            // loop over the (up to) 8 valid kernel positions
            for (int ikd = 0; ikd < num_kd; ++ikd) {
                int kd = kd_arr[ikd];
                int id = (d + padding - kd) / stride;
                if (id < 0 || id >= D_in) continue;

                for (int ikh = 0; ikh < num_kh; ++ikh) {
                    int kh = kh_arr[ikh];
                    int ih = (h + padding - kh) / stride;
                    if (ih < 0 || ih >= H_in) continue;

                    for (int ikw = 0; ikw < num_kw; ++ikw) {
                        int kw = kw_arr[ikw];
                        int iw = (w + padding - kw) / stride;
                        if (iw < 0 || iw >= W_in) continue;

                        // input value – read-only cache
                        float inp = __ldg(in_ptr + ((size_t)id * H_in + ih) * W_in + iw);
                        // weight value – read-only cache
                        float wgt = __ldg(w_ptr + ((size_t)kd * Kh + kh) * Kw + kw);
                        conv_sum += wgt * inp;
                    }
                }
            }
        }

        // ----- add the transposed-conv bias (if any) -----
        conv_sum += __ldg(&conv_bias[c]);

        // ----- fused element-wise op: ((x+bias2)+x)*x+x -----
        float x = conv_sum;
        float b = __ldg(&bias2[c]);
        float tmp  = x + b;          // x + bias2
        float tmp2 = tmp + x;        // (x + bias2) + x  == 2*x + bias2
        float result = tmp2 * x + x; // ((2*x + bias2) * x) + x

        // ----- write final output -----
        output[idx] = result;
    }
}

// Host wrapper that computes output sizes and launches the kernel
torch::Tensor fused_transpose_conv(
    const torch::Tensor& input,       // (N, C_in, D_in, H_in, W_in)
    const torch::Tensor& weight,      // (C_in, C_out, Kd, Kh, Kw)
    const torch::Tensor& conv_bias,   // (C_out) – may be all zeros
    const torch::Tensor& bias2,       // (C_out) – bias for the fused op
    int stride,
    int padding,
    int output_padding,
    int dilation)
{
    const int N = (int)input.size(0);
    const int C_in = (int)input.size(1);
    const int D_in = (int)input.size(2);
    const int H_in = (int)input.size(3);
    const int W_in = (int)input.size(4);

    const int C_out = (int)weight.size(1);
    const int Kd = (int)weight.size(2);
    const int Kh = (int)weight.size(3);
    const int Kw = (int)weight.size(4);

    // output spatial dimensions (formula for transposed conv)
    const int D_out = (D_in - 1) * stride - 2 * padding + dilation * (Kd - 1) + output_padding + 1;
    const int H_out = (H_in - 1) * stride - 2 * padding + dilation * (Kh - 1) + output_padding + 1;
    const int W_out = (W_in - 1) * stride - 2 * padding + dilation * (Kw - 1) + output_padding + 1;

    // allocate result tensor
    torch::Tensor output = torch::empty({N, C_out, D_out, H_out, W_out}, input.options());

    const long long total = (long long)N * C_out * D_out * H_out * W_out;
    const int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    fused_transpose_conv_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        bias2.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        stride, padding, output_padding, dilation,
        Kd, Kh, Kw);

    return output;
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

torch::Tensor fused_transpose_conv(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& bias2,
    int stride,
    int padding,
    int output_padding,
    int dilation);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_transpose_conv", &fused_transpose_conv,
          "Fused transposed 3D convolution + element-wise fused op");
}
"""

# -------------------------------------------------------------------------
# Build the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_transpose_conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional model used for evaluation
# -------------------------------------------------------------------------
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
    bias,
):
    # ---- prepare bias tensors (conv bias may be None) ----
    out_channels = conv_transpose_weight.shape[1]
    if conv_transpose_bias is not None:
        conv_bias = conv_transpose_bias
    else:
        conv_bias = torch.zeros(out_channels, dtype=torch.float32, device='cuda')

    # bias for the fused element-wise part (flatten to 1-D)
    bias_flat = bias.view(-1)

    # ---- launch the fused kernel ----
    return fused_ext.fused_transpose_conv(
        x,
        conv_transpose_weight,
        conv_bias,
        bias_flat,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_dilation
    )

# -------------------------------------------------------------------------
# Helper functions required by the evaluation harness
# -------------------------------------------------------------------------
batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
