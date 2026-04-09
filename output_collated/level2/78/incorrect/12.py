# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_031947/code_3.py
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
#  CUDA source – two kernels and their host wrappers
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>

// ---------------------------------------------------------------
// 1) fused conv_transpose + channel‑wise sum (single channel out)
// ---------------------------------------------------------------
__global__ void conv_transpose_sum_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
    const int C_in,
    const int D_in,
    const int H_in,
    const int W_in,
    const int C_out,
    const int K,
    const int stride,
    const int padding,
    const int D_out,
    const int H_out,
    const int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * D_out * H_out * W_out;
    if (idx >= total) return;

    // decode linear index to (b, d, h, w)
    int tmp = idx;
    int b = tmp / (D_out * H_out * W_out);
    tmp %= (D_out * H_out * W_out);
    int d = tmp / (H_out * W_out);
    tmp %= (H_out * W_out);
    int h = tmp / W_out;
    int w = tmp % W_out;

    float sum_val = 0.0f;

    // loop over output channels
    for (int c_out = 0; c_out < C_out; ++c_out) {
        float acc = 0.0f;

        // loop over input channels and kernel
        for (int c_in = 0; c_in < C_in; ++c_in) {
            int kernel_volume = K * K * K;
            int weight_base = (c_in * C_out + c_out) * kernel_volume;

            for (int kd = 0; kd < K; ++kd) {
                int d_in = d + padding - kd;
                if (d_in < 0) continue;
                if (d_in % stride != 0) continue;
                d_in /= stride;
                if (d_in >= D_in) continue;

                for (int kh = 0; kh < K; ++kh) {
                    int h_in = h + padding - kh;
                    if (h_in < 0) continue;
                    if (h_in % stride != 0) continue;
                    h_in /= stride;
                    if (h_in >= H_in) continue;

                    for (int kw = 0; kw < K; ++kw) {
                        int w_in = w + padding - kw;
                        if (w_in < 0) continue;
                        if (w_in % stride != 0) continue;
                        w_in /= stride;
                        if (w_in >= W_in) continue;

                        // input value
                        int in_idx = ((b * C_in + c_in) * D_in + d_in) * (H_in * W_in)
                                   + (h_in * W_in + w_in);
                        float in_val = input[in_idx];

                        // weight value (no flip required for transposed conv)
                        int kw_idx = ((kd * K) + kh) * K + kw;
                        float w_val = weight[weight_base + kw_idx];

                        acc += in_val * w_val;
                    }
                }
            }
        }

        // add bias for this output channel if provided
        if (bias != nullptr) acc += bias[c_out];
        sum_val += acc;
    }

    output[idx] = sum_val;
}

// ---------------------------------------------------------------
// 2) 3‑D max‑pooling kernel
// ---------------------------------------------------------------
__global__ void max_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch,
    const int C,
    const int D_in,
    const int H_in,
    const int W_in,
    const int K,
    const int stride,
    const int padding,
    const int dilation,
    const int D_out,
    const int H_out,
    const int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * C * D_out * H_out * W_out;
    if (idx >= total) return;

    int tmp = idx;
    int b = tmp / (C * D_out * H_out * W_out);
    tmp %= (C * D_out * H_out * W_out);
    int c = tmp / (D_out * H_out * W_out);
    tmp %= (D_out * H_out * W_out);
    int d = tmp / (H_out * W_out);
    tmp %= (H_out * W_out);
    int h = tmp / W_out;
    int w = tmp % W_out;

    int d_start = d * stride - padding;
    int h_start = h * stride - padding;
    int w_start = w * stride - padding;

    float max_val = -1e38f;   // negative infinity

    for (int kd = 0; kd < K; ++kd) {
        int d_in = d_start + kd * dilation;
        if (d_in < 0 || d_in >= D_in) continue;
        for (int kh = 0; kh < K; ++kh) {
            int h_in = h_start + kh * dilation;
            if (h_in < 0 || h_in >= H_in) continue;
            for (int kw = 0; kw < K; ++kw) {
                int w_in = w_start + kw * dilation;
                if (w_in < 0 || w_in >= W_in) continue;
                int in_idx = ((b * C + c) * D_in + d_in) * (H_in * W_in)
                           + (h_in * W_in + w_in);
                float val = input[in_idx];
                if (val > max_val) max_val = val;
            }
        }
    }
    output[idx] = max_val;
}

// ---------------------------------------------------------------
// Host wrappers (exported to Python)
// ---------------------------------------------------------------
void conv_transpose_sum(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch,
    int C_in,
    int D_in,
    int H_in,
    int W_in,
    int C_out,
    int K,
    int stride,
    int padding,
    int D_out,
    int H_out,
    int W_out,
    torch::Tensor output)
{
    const int threads = 256;
    int total = batch * D_out * H_out * W_out;
    int blocks = (total + threads - 1) / threads;

    const float* bias_ptr = bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;

    conv_transpose_sum_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch, C_in, D_in, H_in, W_in,
        C_out, K, stride, padding,
        D_out, H_out, W_out);
    cudaDeviceSynchronize();
}

void max_pool3d(
    torch::Tensor input,
    int batch,
    int C,
    int D_in,
    int H_in,
    int W_in,
    int K,
    int stride,
    int padding,
    int dilation,
    int D_out,
    int H_out,
    int W_out,
    torch::Tensor output)
{
    const int threads = 256;
    int total = batch * C * D_out * H_out * W_out;
    int blocks = (total + threads - 1) / threads;

    max_pool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, C, D_in, H_in, W_in,
        K, stride, padding, dilation,
        D_out, H_out, W_out);
    cudaDeviceSynchronize();
}
"""

# -------------------------------------------------------------------------
#  C++ interface (PyBind11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void conv_transpose_sum(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch,
    int C_in,
    int D_in,
    int H_in,
    int W_in,
    int C_out,
    int K,
    int stride,
    int padding,
    int D_out,
    int H_out,
    int W_out,
    torch::Tensor output);

void max_pool3d(
    torch::Tensor input,
    int batch,
    int C,
    int D_in,
    int H_in,
    int W_in,
    int K,
    int stride,
    int padding,
    int dilation,
    int D_out,
    int H_out,
    int W_out,
    torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose_sum", &conv_transpose_sum, "Fused conv_transpose + channel sum");
    m.def("max_pool3d", &max_pool3d, "3‑D max pooling");
}
"""

# -------------------------------------------------------------------------
#  Compile the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
#  Helper functions for output size calculations
# -------------------------------------------------------------------------
def conv_transpose_output_size(in_size, kernel_size, stride, padding,
                               output_padding, dilation):
    return (in_size - 1) * stride - 2 * padding + \
           dilation * (kernel_size - 1) + output_padding + 1


def pool_output_size(in_size, kernel_size, stride, padding, dilation, ceil_mode):
    if ceil_mode:
        numerator = in_size + 2 * padding - dilation * (kernel_size - 1) - 1
        return (numerator + stride - 1) // stride + 1
    else:
        return (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


# -------------------------------------------------------------------------
#  functional_model – the only symbol that will be imported
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
    # -----------------------------------------------------------------
    #  Only groups=1 and dilation=1 are implemented – raise otherwise
    # -----------------------------------------------------------------
    if conv_transpose_groups != 1 or conv_transpose_dilation != 1:
        raise NotImplementedError("Only groups=1 and dilation=1 are supported")

    # Ensure bias is a valid tensor (empty tensor if not supplied)
    bias = conv_transpose_bias if conv_transpose_bias is not None \
           else torch.empty(0, dtype=torch.float32, device=x.device)

    batch = x.size(0)
    C_in = x.size(1)
    D_in = x.size(2)
    H_in = x.size(3)
    W_in = x.size(4)

    K = conv_transpose_weight.size(2)            # cubic kernel
    C_out = conv_transpose_weight.size(1)        # weight shape (C_in, C_out, K, K, K)

    # -----------------------------------------------------------
    # 1) Conv_transpose + sum (single‑channel output)
    # -----------------------------------------------------------
    D_out = conv_transpose_output_size(D_in, K, conv_transpose_stride,
                                       conv_transpose_padding,
                                       conv_transpose_output_padding,
                                       conv_transpose_dilation)
    H_out = conv_transpose_output_size(H_in, K, conv_transpose_stride,
                                       conv_transpose_padding,
                                       conv_transpose_output_padding,
                                       conv_transpose_dilation)
    W_out = conv_transpose_output_size(W_in, K, conv_transpose_stride,
                                       conv_transpose_padding,
                                       conv_transpose_output_padding,
                                       conv_transpose_dilation)

    conv_sum_out = torch.empty((batch, 1, D_out, H_out, W_out),
                               dtype=torch.float32, device=x.device)

    fused_ext.conv_transpose_sum(
        x,
        conv_transpose_weight,
        bias,
        batch,
        C_in,
        D_in,
        H_in,
        W_in,
        C_out,
        K,
        conv_transpose_stride,
        conv_transpose_padding,
        D_out,
        H_out,
        W_out,
        conv_sum_out
    )

    # -----------------------------------------------------------
    # 2) First max‑pooling
    # -----------------------------------------------------------
    D1 = pool_output_size(D_out, max_pool1_kernel_size, max_pool1_stride,
                          max_pool1_padding, max_pool1_dilation,
                          max_pool1_ceil_mode)
    H1 = pool_output_size(H_out, max_pool1_kernel_size, max_pool1_stride,
                          max_pool1_padding, max_pool1_dilation,
                          max_pool1_ceil_mode)
    W1 = pool_output_size(W_out, max_pool1_kernel_size, max_pool1_stride,
                          max_pool1_padding, max_pool1_dilation,
                          max_pool1_ceil_mode)

    pool1_out = torch.empty((batch, 1, D1, H1, W1),
                            dtype=torch.float32, device=x.device)

    fused_ext.max_pool3d(
        conv_sum_out,
        batch,
        1,
        D_out,
        H_out,
        W_out,
        max_pool1_kernel_size,
        max_pool1_stride,
        max_pool1_padding,
        max_pool1_dilation,
        D1,
        H1,
        W1,
        pool1_out
    )

    # -----------------------------------------------------------
    # 3) Second max‑pooling
    # -----------------------------------------------------------
    D2 = pool_output_size(D1, max_pool2_kernel_size, max_pool2_stride,
                          max_pool2_padding, max_pool2_dilation,
                          max_pool2_ceil_mode)
    H2 = pool_output_size(H1, max_pool2_kernel_size, max_pool2_stride,
                          max_pool2_padding, max_pool2_dilation,
                          max_pool2_ceil_mode)
    W2 = pool_output_size(W1, max_pool2_kernel_size, max_pool2_stride,
                          max_pool2_padding, max_pool2_dilation,
                          max_pool2_ceil_mode)

    pool2_out = torch.empty((batch, 1, D2, H2, W2),
                            dtype=torch.float32, device=x.device)

    fused_ext.max_pool3d(
        pool1_out,
        batch,
        1,
        D1,
        H1,
        W1,
        max_pool2_kernel_size,
        max_pool2_stride,
        max_pool2_padding,
        max_pool2_dilation,
        D2,
        H2,
        W2,
        pool2_out
    )

    # pool2_out already contains a single channel (the sum)
    return pool2_out


# -------------------------------------------------------------------------
#  Helper functions required by the evaluation harness
# -------------------------------------------------------------------------
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
