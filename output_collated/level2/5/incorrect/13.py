# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_113642/code_6.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'bias_shape', 'stride', 'padding', 'output_padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
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
    # State for conv_transpose (nn.ConvTranspose2d)
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
# CUDA source: custom transposed‑convolution kernel fused with bias + tanh
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ----------------------------------------------------------------------
// Fused transposed convolution kernel
// ----------------------------------------------------------------------
__global__ void deconv_fused_kernel(
    const float* __restrict__ input,   // (N, C_in, H_in, W_in)
    const float* __restrict__ weight, // (C_in, C_out, K_h, K_w)
    const float* __restrict__ bias,   // (C_out,)
    float* __restrict__ output,       // (N, C_out, H_out, W_out)
    const int N, const int C_in, const int C_out,
    const int H_in, const int W_in,
    const int H_out, const int W_out,
    const int K_h, const int K_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int out_pad_h, const int out_pad_w,
    const int dilation_h, const int dilation_w,
    const int groups)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    // ---- decode flat index ------------------------------------------------
    const int c_out = idx % C_out;
    const int w_out = (idx / C_out) % W_out;
    const int h_out = (idx / (C_out * W_out)) % H_out;
    const int n    = idx / (C_out * W_out * H_out);

    // ---- accumulate convolution -----------------------------------------
    float acc = 0.0f;

    // group handling
    const int group_size_out = C_out / groups;
    const int group_id       = c_out / group_size_out;
    const int c_in_start     = group_id * (C_in / groups);
    const int c_in_end       = c_in_start + (C_in / groups);

    // loop over input channels that belong to the same group
    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        // loop over kernel spatial positions
        for (int kh = 0; kh < K_h; ++kh) {
            // compute the y‑coordinate in the input tensor
            int y_in = (h_out + pad_h - kh * dilation_h);
            if (y_in % stride_h != 0) continue;
            y_in /= stride_h;
            if (y_in < 0 || y_in >= H_in) continue;

            for (int kw = 0; kw < K_w; ++kw) {
                // compute the x‑coordinate in the input tensor
                int x_in = (w_out + pad_w - kw * dilation_w);
                if (x_in % stride_w != 0) continue;
                x_in /= stride_w;
                if (x_in < 0 || x_in >= W_in) continue;

                // weight index (flattened as C_in*C_out*K_h*K_w)
                int w_idx = ((c_in * C_out + c_out) * K_h + kh) * K_w + kw;
                float w_val = __ldg(&weight[w_idx]);

                // input index
                int in_idx = ((n * C_in + c_in) * H_in + y_in) * W_in + x_in;
                float in_val = __ldg(&input[in_idx]);

                acc += in_val * w_val;
            }
        }
    }

    // ---- subtract bias ---------------------------------------------------
    acc -= __ldg(&bias[c_out]);

    // ---- apply tanh (fast math) -----------------------------------------
    acc = tanhf(acc);

    // ---- write result ----------------------------------------------------
    output[idx] = acc;
}

// ----------------------------------------------------------------------
// Host function that launches the kernel
// ----------------------------------------------------------------------
void deconv_fused(
    at::Tensor input,   // (N, C_in, H_in, W_in)
    at::Tensor weight, // (C_in, C_out, K_h, K_w)
    at::Tensor bias,   // (C_out,)
    at::Tensor output, // (N, C_out, H_out, W_out) – pre‑allocated
    int N, int C_in, int C_out,
    int H_in, int W_in,
    int H_out, int W_out,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_pad_h, int out_pad_w,
    int dilation_h, int dilation_w,
    int groups)
{
    // ensure data is contiguous and on the GPU
    input   = input.contiguous();
    weight  = weight.contiguous();
    bias    = bias.contiguous();
    output  = output.contiguous();

    const int total = N * C_out * H_out * W_out;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    deconv_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out,
        H_in, W_in,
        H_out, W_out,
        K_h, K_w,
        stride_h, stride_w,
        pad_h, pad_w,
        out_pad_h, out_pad_w,
        dilation_h, dilation_w,
        groups);

    // cudaDeviceSynchronize();  // optional, useful for debugging
}
"""

# ----------------------------------------------------------------------
# C++ bindings (PYBIND11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void deconv_fused(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor output,
    int N, int C_in, int C_out,
    int H_in, int W_in,
    int H_out, int W_out,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_pad_h, int out_pad_w,
    int dilation_h, int dilation_w,
    int groups);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("deconv_fused", &deconv_fused,
          "Fused transposed convolution + bias subtraction + tanh");
}
"""

# ----------------------------------------------------------------------
# Compile the extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_deconv',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True)

# ----------------------------------------------------------------------
# The functional model that will be evaluated
# ----------------------------------------------------------------------
def functional_model(
    x,                      # input tensor (N, C_in, H, W)
    *,
    conv_transpose_weight,  # weight tensor (C_in, C_out, K_h, K_w)
    conv_transpose_bias,    # bias tensor for the convolution (optional, can be None)
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    bias,                   # bias to be subtracted and fed to tanh
):
    # ------------------------------------------------------------------
    # Ensure all tensors are on the GPU and contiguous
    # ------------------------------------------------------------------
    x = x.cuda().contiguous()
    conv_transpose_weight = conv_transpose_weight.cuda().contiguous()
    if conv_transpose_bias is not None:
        conv_transpose_bias = conv_transpose_bias.cuda().contiguous()
    bias = bias.cuda().contiguous()

    N      = x.size(0)
    C_in   = x.size(1)
    H_in   = x.size(2)
    W_in   = x.size(3)

    C_out  = conv_transpose_weight.size(1)
    K_h    = conv_transpose_weight.size(2)
    K_w    = conv_transpose_weight.size(3)

    # ------------------------------------------------------------------
    # Unpack / normalise stride, padding, dilation, output_padding
    # ------------------------------------------------------------------
    if isinstance(conv_transpose_stride, int):
        stride_h = stride_w = conv_transpose_stride
    else:
        stride_h, stride_w = conv_transpose_stride

    if isinstance(conv_transpose_padding, int):
        pad_h = pad_w = conv_transpose_padding
    else:
        pad_h, pad_w = conv_transpose_padding

    if isinstance(conv_transpose_output_padding, int):
        out_pad_h = out_pad_w = conv_transpose_output_padding
    else:
        out_pad_h, out_pad_w = conv_transpose_output_padding

    if isinstance(conv_transpose_dilation, int):
        dil_h = dil_w = conv_transpose_dilation
    else:
        dil_h, dil_w = conv_transpose_dilation

    groups = conv_transpose_groups if conv_transpose_groups is not None else 1

    # ------------------------------------------------------------------
    # Compute output spatial size using the deconvolution formula
    # ------------------------------------------------------------------
    H_out = (H_in - 1) * stride_h - 2 * pad_h + dil_h * (K_h - 1) + out_pad_h + 1
    W_out = (W_in - 1) * stride_w - 2 * pad_w + dil_w * (K_w - 1) + out_pad_w + 1

    # ------------------------------------------------------------------
    # Allocate output tensor
    # ------------------------------------------------------------------
    output = torch.empty(N, C_out, H_out, W_out, dtype=x.dtype, device='cuda')

    # ------------------------------------------------------------------
    # Flatten the post‑conv bias (the original code does view(-1))
    # ------------------------------------------------------------------
    bias_flat = bias.view(-1)

    # ------------------------------------------------------------------
    # Call the fused custom kernel
    # ------------------------------------------------------------------
    fused_ext.deconv_fused(
        x,
        conv_transpose_weight,
        bias_flat,
        output,
        N, C_in, C_out,
        H_in, W_in,
        H_out, W_out,
        K_h, K_w,
        stride_h, stride_w,
        pad_h, pad_w,
        out_pad_h, out_pad_w,
        dil_h, dil_w,
        groups)

    return output
