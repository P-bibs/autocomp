# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_093251/code_3.py
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
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# Inline CUDA source – fused conv_transpose3d + softmax + sigmoid
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

__global__ void fused_conv_softmax_sigmoid_kernel(
    const float* __restrict__ input,      // (batch, in_ch, D_in, H_in, W_in)
    const float* __restrict__ weight,     // (in_ch, out_ch, K, K, K)
    const float* __restrict__ bias,       // (out_ch,) – may be nullptr
    float* __restrict__ output,           // (batch, out_ch, D_out, H_out, W_out)
    const int batch,
    const int in_ch,
    const int out_ch,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int stride,
    const int padding,
    const int output_padding,
    const int dilation,
    const int kernel_size,
    const bool bias_present)               // 1 if bias is valid, 0 otherwise
{
    // ------------------------------------------------------------------
    // Map block index to batch idx and spatial coordinates (od,oh,ow)
    // ------------------------------------------------------------------
    const int total_spatial = D_out * H_out * W_out;
    const int block_id = blockIdx.x;
    const int b = block_id / total_spatial;
    const int spatial_id = block_id % total_spatial;

    int ow = spatial_id % W_out;
    int tmp = spatial_id / W_out;
    int oh = tmp % H_out;
    tmp = tmp / H_out;
    int od = tmp % D_out;

    const int oc = threadIdx.x;                 // output channel for this thread

    if (b >= batch || oc >= out_ch) return;

    // ------------------------------------------------------------------
    // 1) Transposed 3‑D convolution (gather form)
    // ------------------------------------------------------------------
    float conv_val = bias_present ? bias[oc] : 0.0f;

    for (int ic = 0; ic < in_ch; ++ic) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            int iD = (od + padding - kd * dilation);
            if (iD % stride != 0) continue;
            iD /= stride;
            if (iD < 0 || iD >= D_in) continue;

            for (int kh = 0; kh < kernel_size; ++kh) {
                int iH = (oh + padding - kh * dilation);
                if (iH % stride != 0) continue;
                iH /= stride;
                if (iH < 0 || iH >= H_in) continue;

                for (int kw = 0; kw < kernel_size; ++kw) {
                    int iW = (ow + padding - kw * dilation);
                    if (iW % stride != 0) continue;
                    iW /= stride;
                    if (iW < 0 || iW >= W_in) continue;

                    // input index – row‑major 5‑D flatten
                    int inp_idx = (((b * in_ch + ic) * D_in + iD) * H_in + iH) * W_in + iW;
                    float in_val = input[inp_idx];

                    // weight index – row‑major 5‑D flatten (in_ch, out_ch, K, K, K)
                    int w_idx = ((((ic * out_ch + oc) * kernel_size + kd) * kernel_size + kh) * kernel_size + kw);
                    float w_val = weight[w_idx];

                    conv_val += in_val * w_val;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 2) Softmax over the channel dimension (dim=1)
    // ------------------------------------------------------------------
    // Use shared memory for per‑thread intermediate values
    extern __shared__ float sdata[];
    sdata[oc] = conv_val;
    __syncthreads();

    // ----- max reduction -----
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (oc < s) sdata[oc] = fmaxf(sdata[oc], sdata[oc + s]);
        __syncthreads();
    }
    if (oc < 32) {
        float m = sdata[oc];
        for (int offset = 16; offset > 0; offset >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, offset));
        if (oc == 0) sdata[0] = m;
    }
    __syncthreads();
    float max_val = sdata[0];
    __syncthreads();

    // ----- exp(conv - max) -----
    float exp_val = expf(conv_val - max_val);
    sdata[oc] = exp_val;
    __syncthreads();

    // ----- sum reduction -----
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (oc < s) sdata[oc] += sdata[oc + s];
        __syncthreads();
    }
    if (oc < 32) {
        float ssum = sdata[oc];
        for (int offset = 16; offset > 0; offset >>= 1)
            ssum += __shfl_xor_sync(0xffffffff, ssum, offset);
        if (oc == 0) sdata[0] = ssum;
    }
    __syncthreads();
    float sum_exp = sdata[0];
    __syncthreads();

    // ----- softmax -----
    float softmax = exp_val / (sum_exp + 1e-8f);

    // ----- sigmoid -----
    float sigmoid = 1.0f / (1.0f + expf(-softmax));

    // ------------------------------------------------------------------
    // 3) Write final result
    // ------------------------------------------------------------------
    int out_idx = (((b * out_ch + oc) * D_out + od) * H_out + oh) * W_out + ow;
    output[out_idx] = sigmoid;
}

// ----------------------------------------------------------------------
// Host wrapper – launches the fused kernel
// ----------------------------------------------------------------------
void fused_op(
    torch::Tensor input,      // (batch, in_ch, D_in, H_in, W_in)
    torch::Tensor weight,     // (in_ch, out_ch, K, K, K)
    torch::Tensor bias,       // (out_ch,) – may be empty
    torch::Tensor output,     // (batch, out_ch, D_out, H_out, W_out)
    int batch, int in_ch, int out_ch,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int stride, int padding, int output_padding,
    int dilation, int kernel_size,
    bool bias_present)
{
    int block_dim = out_ch;                     // one thread per output channel
    int grid_dim = batch * D_out * H_out * W_out;
    int shared_mem = block_dim * sizeof(float); // shared memory for reductions

    fused_conv_softmax_sigmoid_kernel<<<grid_dim, block_dim, shared_mem>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_present ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch, in_ch, out_ch,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        stride, padding, output_padding,
        dilation, kernel_size,
        bias_present);
}
"""

# ----------------------------------------------------------------------
# C++ binding (PYBIND11) – creates the Python callable "fused_op"
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch, int in_ch, int out_ch,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int stride, int padding, int output_padding,
    int dilation, int kernel_size,
    bool bias_present);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op,
          "Fused transposed 3‑D convolution + softmax + sigmoid");
}
"""

# ----------------------------------------------------------------------
# Compile the extension (one‑time, on‑the‑fly)
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional model that will be imported / evaluated
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
    """
    Fused CUDA implementation of:
        conv_transpose3d -> softmax(dim=1) -> sigmoid
    The implementation assumes groups==1 and softmax over the channel dimension.
    """
    # ------------------------------------------------------------------
    # Move data to GPU
    # ------------------------------------------------------------------
    x = x.cuda()
    weight = conv_transpose_weight.cuda()
    bias = conv_transpose_bias.cuda() if conv_transpose_bias is not None else torch.empty(0, dtype=x.dtype, device='cuda')
    bias_present = conv_transpose_bias is not None and conv_transpose_bias.numel() > 0

    # ------------------------------------------------------------------
    # Extract shapes
    # ------------------------------------------------------------------
    batch, in_ch, D_in, H_in, W_in = x.shape
    out_ch = weight.shape[1]                     # weight is (in_ch, out_ch, K, K, K)
    kernel_size = weight.shape[2]                # cubic kernel assumed

    # ------------------------------------------------------------------
    # Compute output spatial size (standard transposed conv formula)
    # ------------------------------------------------------------------
    stride = conv_transpose_stride
    padding = conv_transpose_padding
    output_padding = conv_transpose_output_padding
    dilation = conv_transpose_dilation

    D_out = (D_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

    # ------------------------------------------------------------------
    # Allocate output tensor
    # ------------------------------------------------------------------
    output = torch.empty((batch, out_ch, D_out, H_out, W_out), dtype=x.dtype, device='cuda')

    # ------------------------------------------------------------------
    # Launch fused kernel
    # ------------------------------------------------------------------
    fused_ext.fused_op(
        x, weight, bias, output,
        batch, in_ch, out_ch,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        stride, padding, output_padding,
        dilation, kernel_size,
        bias_present
    )
    return output


# ----------------------------------------------------------------------
# Helper functions (mimic the original interface, not strictly required)
# ----------------------------------------------------------------------
def get_init_inputs():
    # [in_channels, out_channels, kernel_size, stride, padding, output_padding]
    return [32, 64, 3, 2, 1, 1]

def get_inputs():
    # Random input tensor with the shape used in the benchmark
    return [torch.rand(16, 32, 16, 32, 32)]
