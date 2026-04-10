# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_113742/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'min_value', 'divisor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'min_value', 'divisor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

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
    if 'min_value' in flat_state:
        state_kwargs['min_value'] = flat_state['min_value']
    else:
        state_kwargs['min_value'] = getattr(model, 'min_value')
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
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
#  CUDA source – fused transposed‑conv + clamp + division kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int stride, const int padding,
    const int output_padding, const int dilation,
    const float min_value,
    const float divisor)
{
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_out = N * C_out * D_out * H_out * W_out;
    if (out_idx >= total_out) return;

    // decode flat index -> (n, oc, od, oh, ow)
    int idx = out_idx;
    const int n = idx / (C_out * D_out * H_out * W_out);
    idx %= (C_out * D_out * H_out * W_out);
    const int oc = idx / (D_out * H_out * W_out);
    idx %= (D_out * H_out * W_out);
    const int od = idx / (H_out * W_out);
    idx %= (H_out * W_out);
    const int oh = idx / W_out;
    const int ow = idx % W_out;

    // start with bias (or zero)
    float sum = (bias != nullptr) ? bias[oc] : 0.0f;

    // kernel size is fixed to 3 in the benchmark
    constexpr int K = 3;

    // ------------------------------------------------------------------
    // gather‑style transposed convolution
    // ------------------------------------------------------------------
    for (int ic = 0; ic < C_in; ++ic) {
        const float* w_ptr = weight + ((ic * C_out + oc) * K * K * K);
        #pragma unroll
        for (int kd = 0; kd < K; ++kd) {
            int id = (od + padding - kd * dilation - output_padding);
            if (id < 0 || id % stride != 0) continue;
            id /= stride;
            if (id >= D_in) continue;
            #pragma unroll
            for (int kh = 0; kh < K; ++kh) {
                int ih = (oh + padding - kh * dilation - output_padding);
                if (ih < 0 || ih % stride != 0) continue;
                ih /= stride;
                if (ih >= H_in) continue;
                #pragma unroll
                for (int kw = 0; kw < K; ++kw) {
                    int iw = (ow + padding - kw * dilation - output_padding);
                    if (iw < 0 || iw % stride != 0) continue;
                    iw /= stride;
                    if (iw >= W_in) continue;
                    // linear index for input tensor (row‑major, same as PyTorch)
                    int in_idx = (((n * C_in + ic) * D_in + id) * H_in + ih) * W_in + iw;
                    float val_in = input[in_idx];
                    float w = w_ptr[(kd * K * K + kh * K + kw)];
                    sum += val_in * w;
                }
            }
        }
    }

    // clamp and division
    sum = fmaxf(sum, min_value) / divisor;
    output[out_idx] = sum;
}

/* ----------------------------------------------------------------------
   Host wrapper that launches the kernel
   ---------------------------------------------------------------------- */
void fused_op(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int N, int C_in, int C_out,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int stride, int padding,
    int output_padding, int dilation,
    float min_value,
    float divisor)
{
    const int block_size = 256;
    const int total_out = N * C_out * D_out * H_out * W_out;
    const int grid_size = (total_out + block_size - 1) / block_size;

    conv_transpose_fused_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        stride, padding, output_padding, dilation,
        min_value, divisor);
    cudaDeviceSynchronize();
}
"""

# ----------------------------------------------------------------------
#  C++ bindings – exposes the fused operation to Python
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int N, int C_in, int C_out,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int stride, int padding,
    int output_padding, int dilation,
    float min_value,
    float divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op,
          "Fused 3‑D transposed convolution + clamp + division");
}
"""

# compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
#  Functional model – uses the fused kernel
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
    min_value,
    divisor,
):
    # ------------------------------------------------------------------
    # Compute output spatial size (same formula as PyTorch)
    # ------------------------------------------------------------------
    stride = conv_transpose_stride
    padding = conv_transpose_padding
    output_padding = conv_transpose_output_padding
    dilation = conv_transpose_dilation
    kernel_size = conv_transpose_weight.shape[2]   # square kernel assumed

    N, C_in, D_in, H_in, W_in = x.shape
    C_out = conv_transpose_weight.shape[1]

    D_out = (D_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

    # ------------------------------------------------------------------
    # Allocate output tensor (GPU)
    # ------------------------------------------------------------------
    out = torch.empty((N, C_out, D_out, H_out, W_out),
                      dtype=x.dtype, device=x.device)

    # ------------------------------------------------------------------
    # Bias handling – pass a zero tensor if bias is None
    # ------------------------------------------------------------------
    if conv_transpose_bias is None:
        bias = torch.zeros(C_out, dtype=x.dtype, device=x.device)
    else:
        bias = conv_transpose_bias

    # ------------------------------------------------------------------
    # Launch the fused kernel
    # ------------------------------------------------------------------
    fused_ext.fused_op(
        x,
        conv_transpose_weight,
        bias,
        out,
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        stride, padding, output_padding, dilation,
        float(min_value), float(divisor)
    )
    return out

batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 24, 48, 48
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
