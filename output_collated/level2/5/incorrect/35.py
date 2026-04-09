# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_115905/code_10.py
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

# -------------------------------------------------------------------------
# CUDA source – the fused transposed‑convolution + bias + tanh kernel
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// -------------------------------------------------------------------
// Fused transposed‑convolution kernel
//   - input   : (N, C_in, H_in, W_in)  (float)
//   - weight  : (C_out, C_in, kH, kW)  (float, row‑major flatten)
//   - conv_bias   : (C_out,)  (float)
//   - act_bias    : (C_out,)  (float)
//   - output  : (N, C_out, H_out, W_out) (float, in‑place)
// -------------------------------------------------------------------
__global__ void fused_transpose_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ act_bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int H_in, const int W_in,
    const int H_out, const int W_out,
    const int kH, const int kW,
    const int stride, const int padding,
    const int output_padding, const int dilation)
{
    // each thread computes one output element
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_out = N * C_out * H_out * W_out;
    if (idx >= total_out) return;

    // ---- decode flat index to (n, co, h_out, w_out) ----
    int rem = idx;
    const int co = rem % C_out;
    rem /= C_out;
    const int h_out = rem % H_out;
    const int w_out = rem / H_out;
    const int n = rem / (H_out * W_out);   // actually not needed in the loop

    // ---- convolution sum ----
    float sum = 0.0f;

    // loops over input channels and kernel positions
    for (int ci = 0; ci < C_in; ++ci) {
        for (int kh = 0; kh < kH; ++kh) {
            // compute the input row index according to the transposed‑conv formula
            int h_in = (h_out + padding - kh * dilation);
            if (h_in < 0 || (h_in % stride) != 0) continue;
            h_in /= stride;
            if (h_in >= H_in) continue;

            for (int kw = 0; kw < kW; ++kw) {
                int w_in = (w_out + padding - kw * dilation);
                if (w_in < 0 || (w_in % stride) != 0) continue;
                w_in /= stride;
                if (w_in >= W_in) continue;

                // weight index (flattened row‑major)
                const int wIdx = ((co * C_in + ci) * kH + kh) * kW + kw;
                const float wVal = __ldg(&weight[wIdx]);

                // input index (flattened row‑major)
                const int iIdx = ((n * C_in + ci) * H_in + h_in) * W_in + w_in;
                const float iVal = __ldg(&input[iIdx]);

                sum += wVal * iVal;
            }
        }
    }

    // ---- add convolution bias ----
    sum += __ldg(&conv_bias[co]);

    // ---- subtract activation bias and apply tanh ----
    sum = tanhf(sum - __ldg(&act_bias[co]));

    // ---- write result ----
    output[idx] = sum;
}

// -------------------------------------------------------------------
// Wrapper that sets up the grid and launches the kernel
// -------------------------------------------------------------------
void fused_transpose_conv(
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& conv_bias,
    torch::Tensor& act_bias,
    torch::Tensor& output,
    int stride, int padding, int output_padding, int dilation)
{
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);

    const int C_out = weight.size(0);
    const int kH = weight.size(2);
    const int kW = weight.size(3);

    const int H_out = (H_in - 1) * stride - 2 * padding + dilation * (kH - 1) + output_padding + 1;
    const int W_out = (W_in - 1) * stride - 2 * padding + dilation * (kW - 1) + output_padding + 1;

    const int total_out = N * C_out * H_out * W_out;
    const int threads = 256;
    const int blocks = (total_out + threads - 1) / threads;

    fused_transpose_conv_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        act_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out,
        H_in, W_in,
        H_out, W_out,
        kH, kW,
        stride, padding, output_padding, dilation);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA kernel launch failed");
    }
}
"""

# -------------------------------------------------------------------------
# C++ binding – expose the wrapper to Python
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_transpose_conv(
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& conv_bias,
    torch::Tensor& act_bias,
    torch::Tensor& output,
    int stride, int padding, int output_padding, int dilation);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_transpose_conv",
          &fused_transpose_conv,
          "Fused transposed convolution + bias + tanh");
}
"""

# -------------------------------------------------------------------------
# Compile the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_transpose_conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Helper to compute output spatial size (same formula as PyTorch)
def conv_transpose_output_size(in_size, stride, padding, kernel_size,
                                dilation, output_padding):
    return (in_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1


# -------------------------------------------------------------------------
# functional_model – the entry point that will be imported
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
    """
    Transposed 2‑D convolution followed by (x - bias) and tanh.
    All three operations are performed in a single custom CUDA kernel.
    """
    # -----------------------------------------------------------------
    # Ensure all tensors are contiguous float32 on the GPU
    # -----------------------------------------------------------------
    x = x.contiguous().cuda()
    w = conv_transpose_weight.contiguous().cuda()
    b_conv = conv_transpose_bias.contiguous().cuda()
    b_act = bias.view(-1).contiguous().cuda()   # activation bias flattened

    # -----------------------------------------------------------------
    # Compute output spatial dimensions
    # -----------------------------------------------------------------
    H_in = x.size(2)
    W_in = x.size(3)
    kH = w.size(2)
    kW = w.size(3)

    H_out = conv_transpose_output_size(
        H_in, conv_transpose_stride, conv_transpose_padding,
        kH, conv_transpose_dilation, conv_transpose_output_padding)
    W_out = conv_transpose_output_size(
        W_in, conv_transpose_stride, conv_transpose_padding,
        kW, conv_transpose_dilation, conv_transpose_output_padding)

    C_out = w.size(0)
    N = x.size(0)

    # -----------------------------------------------------------------
    # Allocate output tensor
    # -----------------------------------------------------------------
    output = torch.empty(N, C_out, H_out, W_out, dtype=torch.float32, device='cuda')

    # -----------------------------------------------------------------
    # Launch the fused kernel
    # -----------------------------------------------------------------
    fused_ext.fused_transpose_conv(
        x, w, b_conv, b_act, output,
        conv_transpose_stride, conv_transpose_padding,
        conv_transpose_output_padding, conv_transpose_dilation)

    return output


# -------------------------------------------------------------------------
# Boiler‑plate – required by the evaluation harness
# -------------------------------------------------------------------------
# Setup for testing
batch_size = 32
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]
