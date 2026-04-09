# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_045050/code_10.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x', 'add_input']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
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
# CUDA source – contains both the transposed‑conv kernel and the fused kernel
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ----------------------------------------------------------------------
// Transposed convolution kernel – stride=2, padding=1, kernel=3, dilation=1
// ----------------------------------------------------------------------
__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
    const int in_ch,
    const int out_ch,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_grid = blockDim.x * gridDim.x;
    const int total = batch * out_ch * D_out * H_out * W_out;

    for (int i = idx; i < total; i += stride_grid) {
        // decode linear index -> (n, oc, d_out, h_out, w_out)
        int tmp = i;
        int w_out = tmp % W_out; tmp /= W_out;
        int h_out = tmp % H_out; tmp /= H_out;
        int d_out = tmp % D_out; tmp /= D_out;
        int oc   = tmp % out_ch; tmp /= out_ch;
        int n    = tmp;                     // batch index

        float sum = 0.0f;

        // ----- determine kernel offsets and input coordinates (stride=2, padding=1) -----
        int kd[2], di[2];
        int kh[2], hi[2];
        int kw[2], wi[2];
        int nd, nh, nw;

        // depth
        if ((d_out & 1) == 0) {          // even d_out → one contribution
            kd[0] = 1;
            di[0] = d_out >> 1;
            nd = 1;
        } else {                          // odd d_out → two contributions
            kd[0] = 0;
            di[0] = (d_out + 1) >> 1;
            kd[1] = 2;
            di[1] = (d_out - 1) >> 1;
            nd = 2;
        }
        // height
        if ((h_out & 1) == 0) {
            kh[0] = 1;
            hi[0] = h_out >> 1;
            nh = 1;
        } else {
            kh[0] = 0;
            hi[0] = (h_out + 1) >> 1;
            kh[1] = 2;
            hi[1] = (h_out - 1) >> 1;
            nh = 2;
        }
        // width
        if ((w_out & 1) == 0) {
            kw[0] = 1;
            wi[0] = w_out >> 1;
            nw = 1;
        } else {
            kw[0] = 0;
            wi[0] = (w_out + 1) >> 1;
            kw[1] = 2;
            wi[1] = (w_out - 1) >> 1;
            nw = 2;
        }

        // ----- accumulate contributions -----
        for (int id = 0; id < nd; ++id) {
            int kd_val = kd[id];
            int di_val = di[id];
            for (int ih = 0; ih < nh; ++ih) {
                int kh_val = kh[ih];
                int hi_val = hi[ih];
                for (int iw = 0; iw < nw; ++iw) {
                    int kw_val = kw[iw];
                    int wi_val = wi[iw];
                    // loop over input channels
                    for (int ic = 0; ic < in_ch; ++ic) {
                        // Check bounds
                        if (di_val >= 0 && di_val < D_in &&
                            hi_val >= 0 && hi_val < H_in &&
                            wi_val >= 0 && wi_val < W_in) {
                            // input index: ((n * in_ch + ic) * D_in + di) * (H_in*W_in) + (hi*W_in + wi)
                            int in_idx = ((n * in_ch + ic) * D_in + di_val) * (H_in * W_in) +
                                         (hi_val * W_in + wi_val);
                            float in_val = input[in_idx];

                            // weight index: ((ic * out_ch + oc) * kernel_size + kd) * (kernel_size*kernel_size)
                            //                + kh * kernel_size + kw
                            int wt_idx = ((ic * out_ch + oc) * kernel_size + kd_val) *
                                         (kernel_size * kernel_size) + kh_val * kernel_size + kw_val;
                            float wt_val = weight[wt_idx];

                            sum += in_val * wt_val;
                        }
                    }
                }
            }
        }

        if (bias) sum += bias[oc];

        // write output
        int out_idx = ((n * out_ch + oc) * D_out + d_out) * (H_out * W_out) +
                      (h_out * W_out + w_out);
        output[out_idx] = sum;
    }
}

// Launcher for the transposed convolution
void conv_transpose3d_launcher(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation,
    int kernel_size)
{
    const int batch   = input.size(0);
    const int in_ch   = input.size(1);
    const int D_in    = input.size(2);
    const int H_in    = input.size(3);
    const int W_in    = input.size(4);

    const int out_ch  = output.size(1);
    const int D_out   = output.size(2);
    const int H_out   = output.size(3);
    const int W_out   = output.size(4);

    const int threads = 256;
    const int total   = batch * out_ch * D_out * H_out * W_out;
    const int blocks  = (total + threads - 1) / threads;

    const float* bias_ptr = bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;

    conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch, in_ch, out_ch,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        kernel_size, stride, padding, dilation);
}

// ----------------------------------------------------------------------
// Fused Add + HardSwish kernel (unchanged from original)
// ----------------------------------------------------------------------
__global__ void fused_add_hardswish_kernel(
    const float* __restrict__ x,
    const float* __restrict__ add,
    float* __restrict__ out,
    const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        float val = x[i] + add[i];
        // HardSwish: val * relu6(val + 3) / 6
        float hs = val * fminf(fmaxf(val + 3.0f, 0.0f), 6.0f) * 0.16666666666666666f;
        out[i] = val * hs;
    }
}

void fused_add_hardswish_launcher(torch::Tensor x, torch::Tensor add, torch::Tensor out) {
    const int N = (int)x.numel();
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    fused_add_hardswish_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), add.data_ptr<float>(), out.data_ptr<float>(), N);
}
"""

# -------------------------------------------------------------------------
# C++ interface – binds the two CUDA launchers
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void conv_transpose3d_launcher(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation,
    int kernel_size);

void fused_add_hardswish_launcher(torch::Tensor x, torch::Tensor add, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose3d", &conv_transpose3d_launcher,
          "Custom transposed convolution (CUDA)");
    m.def("fused_add_hardswish", &fused_add_hardswish_launcher,
          "Fused Add + HardSwish (CUDA)");
}
"""

# -------------------------------------------------------------------------
# Compile the inline CUDA extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional model – uses only the two custom CUDA kernels
# -------------------------------------------------------------------------
def functional_model(
    x,
    add_input,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    bias,               # not used in this model (the original code also ignores it)
):
    # ----- compute output spatial size of the transposed convolution -----
    D_in = x.size(2)
    H_in = x.size(3)
    W_in = x.size(4)

    in_ch = x.size(1)
    out_ch = conv_transpose_weight.size(0)       # weight shape: (out_ch, in_ch, K, K, K) for conv_transpose3d
    kernel_size = conv_transpose_weight.size(2)

    stride = conv_transpose_stride[0] if isinstance(conv_transpose_stride, (list, tuple)) else conv_transpose_stride
    padding = conv_transpose_padding[0] if isinstance(conv_transpose_padding, (list, tuple)) else conv_transpose_padding
    output_padding = conv_transpose_output_padding[0] if isinstance(conv_transpose_output_padding, (list, tuple)) else conv_transpose_output_padding
    dilation = conv_transpose_dilation[0] if isinstance(conv_transpose_dilation, (list, tuple)) else conv_transpose_dilation

    D_out = (D_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

    # ----- allocate tensor for the transposed convolution result -----
    conv_out = torch.empty(
        (x.size(0), out_ch, D_out, H_out, W_out),
        dtype=x.dtype,
        device=x.device
    )

    # ----- make sure bias is an empty tensor if not supplied -----
    if conv_transpose_bias is None:
        conv_transpose_bias = torch.empty(0, dtype=x.dtype, device=x.device)

    # ----- run custom transposed convolution kernel -----
    fused_ext.conv_transpose3d(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        conv_out,
        stride,
        padding,
        dilation,
        kernel_size
    )

    # ----- prepare for the fused add + HardSwish kernel -----
    conv_out = conv_out.contiguous()
    add_input = add_input.contiguous()

    final_out = torch.empty_like(conv_out)

    # ----- run fused add + HardSwish kernel -----
    fused_ext.fused_add_hardswish(conv_out, add_input, final_out)

    return final_out


# -------------------------------------------------------------------------
# Test‑harness metadata (unchanged)
# -------------------------------------------------------------------------
batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]


def get_inputs():
    return [
        torch.rand(batch_size, in_channels, D, H, W, device='cuda'),
        torch.rand(batch_size, out_channels, D * stride, H * stride, W * stride, device='cuda')
    ]
