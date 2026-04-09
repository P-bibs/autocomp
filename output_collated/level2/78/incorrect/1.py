# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_025443/code_3.py
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

# -------------------------------------------------------------------------
#  Inline CUDA source (kernels + host wrappers)
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

/* ---------------------------------------------------------------
 * 1) Deconvolution (conv_transpose3d) – one thread per output element
 * --------------------------------------------------------------- */
__global__ void conv_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int K, const int stride,
    const int padding, const int dilation,
    const int groups)                     // groups is ignored (only group=1 used)
{
    int total = N * C_out * D_out * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int tmp   = idx / W_out;
    int h_out = tmp % H_out;
    tmp       = tmp / H_out;
    int d_out = tmp % D_out;
    tmp       = tmp / D_out;
    int c_out = tmp % C_out;
    int n     = tmp / C_out;

    float sum = 0.0f;

    // loop over input channels and kernel spatial size
    for (int c_in = 0; c_in < C_in; ++c_in) {
        int weight_base = ((c_in * C_out + c_out) * K);
        for (int kd = 0; kd < K; ++kd) {
            int d_in_tmp = d_out + padding - kd * dilation;
            if (d_in_tmp < 0) continue;
            if (d_in_tmp % stride != 0) continue;
            int d_in = d_in_tmp / stride;
            if (d_in >= D_in) continue;

            for (int kh = 0; kh < K; ++kh) {
                int h_in_tmp = h_out + padding - kh * dilation;
                if (h_in_tmp < 0) continue;
                if (h_in_tmp % stride != 0) continue;
                int h_in = h_in_tmp / stride;
                if (h_in >= H_in) continue;

                for (int kw = 0; kw < K; ++kw) {
                    int w_in_tmp = w_out + padding - kw * dilation;
                    if (w_in_tmp < 0) continue;
                    if (w_in_tmp % stride != 0) continue;
                    int w_in = w_in_tmp / stride;
                    if (w_in >= W_in) continue;

                    // input value
                    int in_idx = ((n * C_in + c_in) * D_in + d_in) * (H_in * W_in) + h_in * W_in + w_in;
                    float inp_val = input[in_idx];

                    // weight value
                    int w_idx = ((weight_base + kd) * K + kh) * K + kw;
                    float w_val = weight[w_idx];

                    sum += inp_val * w_val;
                }
            }
        }
    }

    if (bias) sum += bias[c_out];
    output[idx] = sum;
}

/* ---------------------------------------------------------------
 * 2) Fused max‑pool (two layers) + channel‑wise sum
 * --------------------------------------------------------------- */
__global__ void pool_sum_kernel(
    const float* __restrict__ conv_out,
    float* __restrict__ out,
    const int N, const int C_out,
    const int D_out, const int H_out, const int W_out,
    const int D1, const int H1, const int W1,
    const int D2, const int H2, const int W2,
    const int k1, const int s1, const int p1, const int d1,
    const int k2, const int s2, const int p2, const int d2)
{
    int total = N * D2 * H2 * W2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int w2   = idx % W2;
    int tmp  = idx / W2;
    int h2   = tmp % H2;
    tmp      = tmp / H2;
    int d2i  = tmp % D2;
    int n    = tmp / D2;

    float sum = 0.0f;

    // iterate over output channels
    for (int c = 0; c < C_out; ++c) {
        float max_val = -1e38f;

        // ---- second pooling layer (k2) ----
        for (int l2d = 0; l2d < k2; ++l2d) {
            int p1_d = d2i * s2 - p2 + l2d * d2;
            if (p1_d < 0 || p1_d >= D1) continue;
            for (int l2h = 0; l2h < k2; ++l2h) {
                int p1_h = h2 * s2 - p2 + l2h * d2;
                if (p1_h < 0 || p1_h >= H1) continue;
                for (int l2w = 0; l2w < k2; ++l2w) {
                    int p1_w = w2 * s2 - p2 + l2w * d2;
                    if (p1_w < 0 || p1_w >= W1) continue;

                    // ---- first pooling layer (k1) ----
                    for (int l1d = 0; l1d < k1; ++l1d) {
                        int d0 = p1_d * s1 - p1 + l1d * d1;
                        if (d0 < 0 || d0 >= D_out) continue;
                        for (int l1h = 0; l1h < k1; ++l1h) {
                            int h0 = p1_h * s1 - p1 + l1h * d1;
                            if (h0 < 0 || h0 >= H_out) continue;
                            for (int l1w = 0; l1w < k1; ++l1w) {
                                int w0 = p1_w * s1 - p1 + l1w * d1;
                                if (w0 < 0 || w0 >= W_out) continue;

                                int idx_conv = ((n * C_out + c) * D_out + d0) * (H_out * W_out) + h0 * W_out + w0;
                                float v = conv_out[idx_conv];
                                if (v > max_val) max_val = v;
                            }
                        }
                    }
                }
            }
        }
        sum += max_val;
    }
    out[idx] = sum;
}

/* ---------------------------------------------------------------
 * Host wrappers (launch the kernels)
 * --------------------------------------------------------------- */
void conv_transpose_launch(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int N, int C_in, int C_out,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int K, int stride,
    int padding, int dilation, int groups)
{
    const float* in_ptr   = input.data_ptr<float>();
    const float* w_ptr    = weight.data_ptr<float>();
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    float* out_ptr        = output.data_ptr<float>();

    int total = N * C_out * D_out * H_out * W_out;
    const int block_sz = 256;
    int grid = (total + block_sz - 1) / block_sz;

    conv_transpose_kernel<<<grid, block_sz>>>(
        in_ptr, w_ptr, bias_ptr, out_ptr,
        N, C_in, C_out, D_in, H_in, W_in,
        D_out, H_out, W_out,
        K, stride, padding, dilation, groups);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("conv_transpose kernel error: %s\n", cudaGetErrorString(err));
}

void pool_sum_launch(
    torch::Tensor conv_out,
    torch::Tensor out,
    int N, int C_out,
    int D_out, int H_out, int W_out,
    int D1, int H1, int W1,
    int D2, int H2, int W2,
    int k1, int s1, int p1, int d1,
    int k2, int s2, int p2, int d2)
{
    const float* in_ptr = conv_out.data_ptr<float>();
    float* out_ptr       = out.data_ptr<float>();

    int total = N * D2 * H2 * W2;
    const int block_sz = 256;
    int grid = (total + block_sz - 1) / block_sz;

    pool_sum_kernel<<<grid, block_sz>>>(
        in_ptr, out_ptr,
        N, C_out, D_out, H_out, W_out,
        D1, H1, W1, D2, H2, W2,
        k1, s1, p1, d1, k2, s2, p2, d2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("pool_sum kernel error: %s\n", cudaGetErrorString(err));
}
"""

# -------------------------------------------------------------------------
#  C++ binding (pybind11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void conv_transpose_launch(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int N, int C_in, int C_out,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int K, int stride,
    int padding, int dilation, int groups);

void pool_sum_launch(
    torch::Tensor conv_out,
    torch::Tensor out,
    int N, int C_out,
    int D_out, int H_out, int W_out,
    int D1, int H1, int W1,
    int D2, int H2, int W2,
    int k1, int s1, int p1, int d1,
    int k2, int s2, int p2, int d2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose", &conv_transpose_launch, "conv_transpose forward");
    m.def("pool_sum", &pool_sum_launch, "fused pool + sum");
}
"""

# -------------------------------------------------------------------------
#  Compile the inline CUDA extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)


# -------------------------------------------------------------------------
#  The functional model that will be imported / evaluated
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
    """
    Fused implementation:
        1) custom conv_transpose3d (single kernel)
        2) fused max‑pool → max‑pool → sum (single kernel)
    The function signature matches the original one; all extra parameters are
    ignored except those that affect the geometry of the operations.
    """

    # -----------------------------------------------------------------
    #  Move tensors to GPU if they are not already there
    # -----------------------------------------------------------------
    if not x.is_cuda:
        x = x.cuda()
    if not conv_transpose_weight.is_cuda:
        conv_transpose_weight = conv_transpose_weight.cuda()
    if conv_transpose_bias is not None and not conv_transpose_bias.is_cuda:
        conv_transpose_bias = conv_transpose_bias.cuda()

    # -----------------------------------------------------------------
    #  Shapes
    # -----------------------------------------------------------------
    N = x.size(0)          # batch size
    C_in = x.size(1)       # input channels
    D_in = x.size(2)
    H_in = x.size(3)
    W_in = x.size(4)

    K = conv_transpose_weight.size(2)          # kernel size (square)
    C_out = conv_transpose_weight.size(1)      # output channels

    stride = conv_transpose_stride
    padding = conv_transpose_padding
    output_padding = conv_transpose_output_padding
    dilation = conv_transpose_dilation

    # -----------------------------------------------------------------
    #  Output size after conv_transpose
    # -----------------------------------------------------------------
    D_out = (D_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1

    # allocate intermediate buffer
    conv_out = torch.empty(N, C_out, D_out, H_out, W_out,
                           dtype=torch.float32, device=x.device)

    # -----------------------------------------------------------------
    #  Kernel 1 : custom conv_transpose3d
    # -----------------------------------------------------------------
    fused_ext.conv_transpose(
        x, conv_transpose_weight, conv_transpose_bias, conv_out,
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        K, stride, padding, dilation, conv_transpose_groups)

    # -----------------------------------------------------------------
    #  Pooling parameters (both layers)
    # -----------------------------------------------------------------
    k1 = max_pool1_kernel_size
    s1 = max_pool1_stride
    p1 = max_pool1_padding
    d1 = max_pool1_dilation

    k2 = max_pool2_kernel_size
    s2 = max_pool2_stride
    p2 = max_pool2_padding
    d2 = max_pool2_dilation

    # size after first pooling
    D1 = (D_out + 2 * p1 - d1 * (k1 - 1) - 1) // s1 + 1
    H1 = (H_out + 2 * p1 - d1 * (k1 - 1) - 1) // s1 + 1
    W1 = (W_out + 2 * p1 - d1 * (k1 - 1) - 1) // s1 + 1

    # size after second pooling (final output)
    D2 = (D1 + 2 * p2 - d2 * (k2 - 1) - 1) // s2 + 1
    H2 = (H1 + 2 * p2 - d2 * (k2 - 1) - 1) // s2 + 1
    W2 = (W1 + 2 * p2 - d2 * (k2 - 1) - 1) // s2 + 1

    # allocate final output (batch, 1, D2, H2, W2)
    final_out = torch.empty(N, 1, D2, H2, W2,
                           dtype=torch.float32, device=x.device)

    # -----------------------------------------------------------------
    #  Kernel 2 : fused max‑pool (twice) + channel‑wise sum
    # -----------------------------------------------------------------
    fused_ext.pool_sum(
        conv_out, final_out,
        N, C_out,
        D_out, H_out, W_out,
        D1, H1, W1,
        D2, H2, W2,
        k1, s1, p1, d1,
        k2, s2, p2, d2)

    return final_out

batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 32, 32, 32
kernel_size = 5
stride = 2
padding = 2

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
