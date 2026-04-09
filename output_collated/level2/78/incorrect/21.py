# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_033345/code_3.py
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
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
#  CUDA source – two fused kernels + launch wrappers
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// -------------------------------------------------------------------------
// Kernel 1: 3‑D transposed convolution (naive im2col style)
// -------------------------------------------------------------------------
__global__ void conv_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int K, const int stride, const int padding,
    const int output_padding, const int dilation)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * D_out * H_out * W_out;
    if (idx >= total) return;

    // decode linear index -> (n, oc, od, oh, ow)
    int rem = idx;
    int n = rem / (C_out * D_out * H_out * W_out);
    rem %= (C_out * D_out * H_out * W_out);
    int oc = rem / (D_out * H_out * W_out);
    rem %= (D_out * H_out * W_out);
    int od = rem / (H_out * W_out);
    rem %= (H_out * W_out);
    int oh = rem / W_out;
    int ow = rem % W_out;

    float sum = (bias != nullptr) ? bias[oc] : 0.0f;

    // naive im2col: loop over input channels and kernel positions
    for (int ic = 0; ic < C_in; ++ic) {
        for (int kd = 0; kd < K; ++kd) {
            int id = (od + padding - kd * dilation) / stride;               // floor division
            if (id < 0 || id >= D_in || (od + padding - kd * dilation) % stride != 0) continue;
            for (int kh = 0; kh < K; ++kh) {
                int ih = (oh + padding - kh * dilation) / stride;
                if (ih < 0 || ih >= H_in || (oh + padding - kh * dilation) % stride != 0) continue;
                for (int kw = 0; kw < K; ++kw) {
                    int iw = (ow + padding - kw * dilation) / stride;
                    if (iw < 0 || iw >= W_in || (ow + padding - kw * dilation) % stride != 0) continue;

                    int w_idx = ((oc * C_in + ic) * K + kd) * K * K + kh * K + kw;
                    int i_idx = ((n * C_in + ic) * D_in + id) * H_in * W_in + ih * W_in + iw;
                    sum += weight[w_idx] * input[i_idx];
                }
            }
        }
    }
    output[idx] = sum;
}

// -------------------------------------------------------------------------
// Kernel 2: two max‑pooling layers followed by a channel reduction (sum)
// -------------------------------------------------------------------------
__global__ void maxpool_sum_kernel(
    const float* __restrict__ conv_out,
    float* __restrict__ out,
    const int N, const int C_out,
    const int D_trans, const int H_trans, const int W_trans,
    const int k1, const int s1, const int p1,
    const int k2, const int s2, const int p2,
    const int D_out, const int H_out, const int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * D_out * H_out * W_out;
    if (idx >= total) return;

    // decode linear index -> (n, d2, h2, w2)
    int rem = idx;
    int n = rem / (D_out * H_out * W_out);
    rem %= (D_out * H_out * W_out);
    int d2 = rem / (H_out * W_out);
    rem %= (H_out * W_out);
    int h2 = rem / W_out;
    int w2 = rem % W_out;

    float sum = 0.0f;

    // loop over output channels, find the max in the two‑stage pooling window
    for (int oc = 0; oc < C_out; ++oc) {
        float max_val = -1e38f;

        // second (outer) pooling stage
        for (int kd2 = 0; kd2 < k2; ++kd2) {
            int d1 = d2 * s2 - p2 + kd2;
            if (d1 < 0 || d1 >= D_trans) continue;
            for (int kh2 = 0; kh2 < k2; ++kh2) {
                int h1 = h2 * s2 - p2 + kh2;
                if (h1 < 0 || h1 >= H_trans) continue;
                for (int kw2 = 0; kw2 < k2; ++kw2) {
                    int w1 = w2 * s2 - p2 + kw2;
                    if (w1 < 0 || w1 >= W_trans) continue;

                    // first (inner) pooling stage
                    for (int kd1 = 0; kd1 < k1; ++kd1) {
                        int d0 = d1 * s1 - p1 + kd1;
                        if (d0 < 0 || d0 >= D_trans) continue;
                        for (int kh1 = 0; kh1 < k1; ++kh1) {
                            int h0 = h1 * s1 - p1 + kh1;
                            if (h0 < 0 || h0 >= H_trans) continue;
                            for (int kw1 = 0; kw1 < k1; ++kw1) {
                                int w0 = w1 * s1 - p1 + kw1;
                                if (w0 < 0 || w0 >= W_trans) continue;

                                int idx_conv = ((n * C_out + oc) * D_trans + d0) *
                                                H_trans * W_trans + h0 * W_trans + w0;
                                float val = conv_out[idx_conv];
                                if (val > max_val) max_val = val;
                            }
                        }
                    }
                }
            }
        }
        sum += max_val;
    }

    // write final result (keepdim → channel dimension = 1)
    int out_idx = ((n * 1 + 0) * D_out + d2) * H_out * W_out + h2 * W_out + w2;
    out[out_idx] = sum;
}

// -------------------------------------------------------------------------
// Launch wrappers (called from Python)
// -------------------------------------------------------------------------
void conv_transposeLauncher(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int N, int C_in, int C_out,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int K, int stride, int padding,
    int output_padding, int dilation)
{
    const int threads = 256;
    int total = N * C_out * D_out * H_out * W_out;
    int blocks = (total + threads - 1) / threads;
    conv_transpose_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        K, stride, padding,
        output_padding, dilation);
    cudaDeviceSynchronize();
}

void maxpool_sumLauncher(
    torch::Tensor conv_out,
    torch::Tensor out,
    int N, int C_out,
    int D_trans, int H_trans, int W_trans,
    int k1, int s1, int p1,
    int k2, int s2, int p2,
    int D_out, int H_out, int W_out)
{
    const int threads = 256;
    int total = N * D_out * H_out * W_out;
    int blocks = (total + threads - 1) / threads;
    maxpool_sum_kernel<<<blocks, threads>>>(
        conv_out.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C_out,
        D_trans, H_trans, W_trans,
        k1, s1, p1,
        k2, s2, p2,
        D_out, H_out, W_out);
    cudaDeviceSynchronize();
}
"""

# ----------------------------------------------------------------------
#  C++ binding – exposes the two launchers to Python
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void conv_transposeLauncher(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int N, int C_in, int C_out,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int K, int stride, int padding,
    int output_padding, int dilation);

void maxpool_sumLauncher(
    torch::Tensor conv_out,
    torch::Tensor out,
    int N, int C_out,
    int D_trans, int H_trans, int W_trans,
    int k1, int s1, int p1,
    int k2, int s2, int p2,
    int D_out, int H_out, int W_out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose", &conv_transposeLauncher, "Fused conv_transpose3d kernel");
    m.def("maxpool_sum",    &maxpool_sumLauncher,    "Fused max‑pool + sum kernel");
}
"""

# ----------------------------------------------------------------------
#  Compile the extension (run once when the module is imported)
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# ----------------------------------------------------------------------
#  The functional model that will be evaluated
# ----------------------------------------------------------------------
def functional_model(
    x: torch.Tensor,
    *,
    conv_transpose_weight: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
    conv_transpose_stride: int,
    conv_transpose_padding: int,
    conv_transpose_output_padding: int,
    conv_transpose_groups: int,
    conv_transpose_dilation: int,
    max_pool1_kernel_size: int,
    max_pool1_stride: int,
    max_pool1_padding: int,
    max_pool1_dilation: int,
    max_pool1_ceil_mode: bool,
    max_pool1_return_indices: bool,
    max_pool2_kernel_size: int,
    max_pool2_stride: int,
    max_pool2_padding: int,
    max_pool2_dilation: int,
    max_pool2_ceil_mode: bool,
    max_pool2_return_indices: bool,
) -> torch.Tensor:
    """
    Fused implementation:
        1) 3‑D transposed convolution (custom CUDA kernel)
        2) Two max‑pooling layers + channel‑wise sum (second custom kernel)

    No PyTorch built‑in convolution or pooling functions are used.
    """

    # ------------------------------------------------------------------
    # Basic shape information
    # ------------------------------------------------------------------
    N, C_in, D_in, H_in, W_in = x.shape                     # batch, in‑channels, depth, height, width
    C_out = conv_transpose_weight.shape[0]                   # out‑channels
    K = conv_transpose_weight.shape[2]                       # square kernel size (assume KxKxK)

    # ------------------------------------------------------------------
    # 1) Compute output size of the transposed convolution
    # ------------------------------------------------------------------
    stride = conv_transpose_stride
    padding = conv_transpose_padding
    output_padding = conv_transpose_output_padding
    dilation = conv_transpose_dilation

    D_out = (D_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1

    # Allocate intermediate buffer (N, C_out, D_out, H_out, W_out)
    conv_out = torch.empty((N, C_out, D_out, H_out, W_out),
                           dtype=torch.float32, device=x.device)

    # Make sure a bias tensor is always passed (empty if not needed)
    if conv_transpose_bias is not None:
        bias = conv_transpose_bias
    else:
        bias = torch.empty(0, dtype=torch.float32, device=x.device)

    # Launch the custom conv_transpose kernel
    fused_ext.conv_transpose(
        x, conv_transpose_weight, bias, conv_out,
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        K, stride, padding, output_padding, dilation,
    )

    # ------------------------------------------------------------------
    # 2) Determine sizes after the two max‑pooling layers
    #    (floor pooling, ceil_mode = False)
    # ------------------------------------------------------------------
    k1, s1, p1 = max_pool1_kernel_size, max_pool1_stride, max_pool1_padding
    # pooling formula: out = floor((in + 2*pad - kernel) / stride) + 1
    D1 = (D_out + 2 * p1 - k1) // s1 + 1
    H1 = (H_out + 2 * p1 - k1) // s1 + 1
    W1 = (W_out + 2 * p1 - k1) // s1 + 1

    k2, s2, p2 = max_pool2_kernel_size, max_pool2_stride, max_pool2_padding
    D2 = (D1 + 2 * p2 - k2) // s2 + 1
    H2 = (H1 + 2 * p2 - k2) // s2 + 1
    W2 = (W1 + 2 * p2 - k2) // s2 + 1

    # Allocate final output tensor (batch, 1, D2, H2, W2)
    out = torch.empty((N, 1, D2, H2, W2),
                      dtype=torch.float32, device=x.device)

    # Launch the fused max‑pool + sum kernel
    fused_ext.maxpool_sum(
        conv_out, out,
        N, C_out,
        D_out, H_out, W_out,
        k1, s1, p1,
        k2, s2, p2,
        D2, H2, W2,
    )

    return out


# ----------------------------------------------------------------------
#  Helper functions required by the benchmark harness (not essential for the
#  functional_model itself, but kept for completeness)
# ----------------------------------------------------------------------
def get_init_inputs():
    # Original problem size
    return [32, 64, 5, 2, 2]

def get_inputs():
    # Random input placed on GPU
    return [torch.rand(16, 32, 32, 32, 32, device='cuda')]
