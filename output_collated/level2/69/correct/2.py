# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_052229/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies HardSwish, and then ReLU.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

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
    # State for conv (nn.Conv2d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
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
# Inline CUDA source – fused convolution + hardswish + relu kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_hardswish_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int H_in, const int W_in,
    const int K,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    const int groups,
    const int H_out, const int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_out = N * C_out * H_out * W_out;
    if (idx >= total_out) return;

    // ---- unpack flat index ------------------------------------------------
    int tmp = idx;
    int n = tmp / (C_out * H_out * W_out);
    tmp %= (C_out * H_out * W_out);
    int oc = tmp / (H_out * W_out);
    tmp %= (H_out * W_out);
    int oh = tmp / W_out;
    int ow = tmp % W_out;

    // ---- grouped convolution ---------------------------------------------
    const int c_out_per_group = C_out / groups;
    const int c_in_per_group  = C_in  / groups;
    const int group_id = oc / c_out_per_group;
    const int ic_start = group_id * c_in_per_group;

    float sum = 0.0f;

    for (int ic = ic_start; ic < ic_start + c_in_per_group; ++ic) {
        for (int kh = 0; kh < K; ++kh) {
            int i_h = oh * stride_h + kh * dilation_h - pad_h;
            if (i_h < 0 || i_h >= H_in) continue;
            for (int kw = 0; kw < K; ++kw) {
                int i_w = ow * stride_w + kw * dilation_w - pad_w;
                if (i_w < 0 || i_w >= W_in) continue;

                // weight index: [C_out][c_in_per_group][K][K] flatten
                int wIdx = ((oc * c_in_per_group) + (ic - ic_start)) * (K * K)
                            + kh * K + kw;
                float w = weight[wIdx];

                // input index: (n, ic, i_h, i_w)
                int iIdx = ((n * C_in + ic) * H_in + i_h) * W_in + i_w;
                float v = input[iIdx];

                sum += v * w;
            }
        }
    }

    if (bias) sum += bias[oc];

    // ---- hardswish --------------------------------------------------------
    float x = sum;
    float hs = x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;

    // ---- ReLU -------------------------------------------------------------
    float y = fmaxf(hs, 0.0f);

    output[idx] = y;
}

/* Wrapper that launches the kernel */
void fused_conv(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor output,
    int N, int C_in, int C_out,
    int H_in, int W_in,
    int K,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups,
    int H_out, int W_out)
{
    const float* in_ptr   = input.data_ptr<float>();
    const float* w_ptr    = weight.data_ptr<float>();
    const float* b_ptr    = bias.defined() ? bias.data_ptr<float>() : nullptr;
    float* out_ptr        = output.data_ptr<float>();

    const int total_out = N * C_out * H_out * W_out;
    const int block_sz  = 256;
    const int grid_sz   = (total_out + block_sz - 1) / block_sz;

    fused_conv_hardswish_relu_kernel<<<grid_sz, block_sz>>>(
        in_ptr, w_ptr, b_ptr, out_ptr,
        N, C_in, C_out, H_in, W_in, K,
        stride_h, stride_w, pad_h, pad_w,
        dilation_h, dilation_w, groups,
        H_out, W_out);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel error: ") + cudaGetErrorString(err));
    }
}
"""

# ----------------------------------------------------------------------
# C++ bindings (PYBIND11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_conv(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor output,
    int N, int C_in, int C_out,
    int H_in, int W_in,
    int K,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups,
    int H_out, int W_out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &fused_conv, "Fused conv + hardswish + relu");
}
"""

# ----------------------------------------------------------------------
# Build the inline extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# functional_model – the only entry point that will be imported
# ----------------------------------------------------------------------
def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
):
    # Make sure tensors are contiguous and live on the GPU
    x = x.contiguous()
    conv_weight = conv_weight.contiguous()
    if conv_bias is not None:
        conv_bias = conv_bias.contiguous()

    # ------------------------------------------------------------------
    # Unpack stride / padding / dilation (int or tuple)
    # ------------------------------------------------------------------
    if isinstance(conv_stride, int):
        stride_h = stride_w = conv_stride
    else:
        stride_h, stride_w = conv_stride

    if isinstance(conv_padding, int):
        pad_h = pad_w = conv_padding
    else:
        pad_h, pad_w = conv_padding

    if isinstance(conv_dilation, int):
        dilation_h = dilation_w = conv_dilation
    else:
        dilation_h, dilation_w = conv_dilation

    # ------------------------------------------------------------------
    # Tensor dimensions
    # ------------------------------------------------------------------
    N, C_in, H_in, W_in = x.shape
    C_out = conv_weight.shape[0]
    K = conv_weight.shape[2]                     # square kernel

    # ------------------------------------------------------------------
    # Output spatial size
    # ------------------------------------------------------------------
    H_out = (H_in + 2 * pad_h - dilation_h * (K - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * pad_w - dilation_w * (K - 1) - 1) // stride_w + 1

    # Allocate output
    output = torch.empty(N, C_out, H_out, W_out, dtype=x.dtype, device=x.device)

    # ------------------------------------------------------------------
    # Launch fused kernel
    # ------------------------------------------------------------------
    fused_ext.fused_conv(
        x, conv_weight, conv_bias, output,
        N, C_in, C_out,
        H_in, W_in,
        K,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        conv_groups,
        H_out, W_out
    )

    return output


# ----------------------------------------------------------------------
# Helper functions (not required for evaluation, but kept for completeness)
# ----------------------------------------------------------------------
def get_init_inputs():
    return [8, 64, 3]

def get_inputs():
    return [torch.rand(128, 8, 128, 128, device='cuda')]
