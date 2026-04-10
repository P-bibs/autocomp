# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_140117/code_13.py
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
# CUDA kernel – fused transposed convolution + bias + element‑wise formula
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias1,      // conv_transpose bias (one per out‑channel)
    const float* __restrict__ bias2,      // final bias (one per out‑channel)
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int stride, const int padding, const int dilation,
    const int K, const int total_out)
{
    // each thread handles up to 4 consecutive output elements (vectorised store)
    const int idx_base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx_base >= total_out) return;

    // pre‑compute strides for fast index decoding
    const int out_channel_size = D_out * H_out * W_out;
    const int batch_stride = C_out * out_channel_size;

    // results for the 4 elements
    float4 result_vec = {0.0f, 0.0f, 0.0f, 0.0f};

    // number of elements this thread really has to process
    const int limit = (total_out - idx_base) >= 4 ? 4 : (total_out - idx_base);

    // ---------------------------------------------------------------------
    // outer loop over the 4 output elements assigned to this thread
    // ---------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (i >= limit) break;

        const int idx_elem = idx_base + i;

        // decode flat index into (n, c_out, d, h, w)
        const int n         = idx_elem / batch_stride;
        const int rem1      = idx_elem % batch_stride;
        const int c_out     = rem1 / out_channel_size;
        const int rem2      = rem1 % out_channel_size;
        const int d         = rem2 / (H_out * W_out);
        const int h         = (rem2 % (H_out * W_out)) / W_out;
        const int w         = rem2 % W_out;

        // -----------------------------------------------------------------
        // 1) accumulate convolution sum
        // -----------------------------------------------------------------
        float sum = 0.0f;

        // preload the two bias values for this output channel
        const float b1 = __ldg(&bias1[c_out]);
        const float b2 = __ldg(&bias2[c_out]);

        // loop over input channels and kernel positions
        for (int c_in = 0; c_in < C_in; ++c_in) {
            // weight base for this input channel and output channel
            const int w_base = ((c_in * C_out + c_out) * K) * K * K;

            for (int kd = 0; kd < K; ++kd) {
                const int d_numer = d + padding - kd * dilation;
                if (d_numer % stride != 0) continue;
                const int d_in = d_numer / stride;
                if (d_in < 0 || d_in >= D_in) continue;

                for (int kh = 0; kh < K; ++kh) {
                    const int h_numer = h + padding - kh * dilation;
                    if (h_numer % stride != 0) continue;
                    const int h_in = h_numer / stride;
                    if (h_in < 0 || h_in >= H_in) continue;

                    for (int kw = 0; kw < K; ++kw) {
                        const int w_numer = w + padding - kw * dilation;
                        if (w_numer % stride != 0) continue;
                        const int w_in = w_numer / stride;
                        if (w_in < 0 || w_in >= W_in) continue;

                        // input index – NCDHW layout
                        const int in_idx = (((n * C_in + c_in) * D_in + d_in) * H_in + h_in) * W_in + w_in;
                        const float x = __ldg(&input[in_idx]);

                        // weight index – (C_in, C_out, K, K, K) layout
                        const int w_idx = w_base + ((kd * K + kh) * K + kw);
                        const float w_val = __ldg(&weight[w_idx]);

                        sum += x * w_val;
                    }
                }
            }
        }

        // -----------------------------------------------------------------
        // 2) add conv_transpose bias (bias1) -> this is the "x" value
        // -----------------------------------------------------------------
        const float x = sum + b1;

        // -----------------------------------------------------------------
        // 3) fused element‑wise formula:
        //    ((x + bias2) + x) * x + x  =  (2*x + bias2) * x + x
        // -----------------------------------------------------------------
        const float res = (2.0f * x + b2) * x + x;

        // store to the result vector
        reinterpret_cast<float*>(&result_vec)[i] = res;
    }

    // -----------------------------------------------------------------
    // 4) vectorised store of the four results
    // -----------------------------------------------------------------
    reinterpret_cast<float4*>(output + idx_base)[0] = result_vec;
}

torch::Tensor conv_transpose_fused(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias1,
    const torch::Tensor& bias2,
    const int stride,
    const int padding,
    const int dilation,
    const int K,
    const int D_out,
    const int H_out,
    const int W_out) {

    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias1.is_contiguous(), "bias1 must be contiguous");
    TORCH_CHECK(bias2.is_contiguous(), "bias2 must be contiguous");

    const int N     = input.size(0);
    const int C_in  = input.size(1);
    const int C_out = weight.size(1);

    auto output = torch::empty({N, C_out, D_out, H_out, W_out}, input.options());

    const int total_out = N * C_out * D_out * H_out * W_out;
    const int threads_per_block = 256;
    const int blocks = (total_out + threads_per_block * 4 - 1) / (threads_per_block * 4);

    conv_transpose_fused_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias1.data_ptr<float>(),
        bias2.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out,
        input.size(2), input.size(3), input.size(4),
        D_out, H_out, W_out,
        stride, padding, dilation,
        K, total_out);

    return output;
}

"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

torch::Tensor conv_transpose_fused(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias1,
    const torch::Tensor& bias2,
    const int stride,
    const int padding,
    const int dilation,
    const int K,
    const int D_out,
    const int H_out,
    const int W_out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose_fused", &conv_transpose_fused,
          "Fused transposed convolution + bias + element‑wise formula");
}
"""

# -------------------------------------------------------------------------
# Build the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='conv_transpose_fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional model – entry point for evaluation
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
    # -----------------------------------------------------------------
    # Compute output spatial size (PyTorch formula for transposed conv)
    # -----------------------------------------------------------------
    stride   = conv_transpose_stride[0]          # assume equal strides in all dims
    padding  = conv_transpose_padding
    output_padding = conv_transpose_output_padding
    dilation = conv_transpose_dilation[0]        # assume equal dilation in all dims
    K = conv_transpose_weight.size(2)            # kernel size (square)

    D_in = x.size(2)
    H_in = x.size(3)
    W_in = x.size(4)

    D_out = (D_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1

    # -----------------------------------------------------------------
    # Flatten the two bias tensors (they are 1‑D after view)
    # -----------------------------------------------------------------
    bias1 = conv_transpose_bias.view(-1)               # conv_transpose bias
    bias2 = bias.view(-1)                              # final bias

    # -----------------------------------------------------------------
    # Call the fused CUDA kernel (no PyTorch conv function)
    # -----------------------------------------------------------------
    return fused_ext.conv_transpose_fused(
        x,
        conv_transpose_weight,
        bias1,
        bias2,
        stride,
        padding,
        dilation,
        K,
        D_out,
        H_out,
        W_out
    )

# -------------------------------------------------------------------------
# Shape parameters used by the evaluation harness (not part of the model)
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
