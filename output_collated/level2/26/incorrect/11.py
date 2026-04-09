# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_040909/code_7.py
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

# ----------------------------------------------------------------------
# CUDA source – fused transposed-conv + add + hardswish
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose_add_hardswish_kernel(
    const float* __restrict__ input,      // (N, C_in, D_in, H_in, W_in)
    const float* __restrict__ weight,     // (C_out, C_in, K, K, K) flattened
    const float* __restrict__ add_input,  // (N, C_out, D_out, H_out, W_out)
    const float* __restrict__ bias,       // (C_out) – may be nullptr
    float*       __restrict__ output,     // (N, C_out, D_out, H_out, W_out)
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int stride, const int padding, const int kernel_size)
{
    const int kernel_vol = kernel_size * kernel_size * kernel_size;
    const int total = N * C_out * D_out * H_out * W_out;

    // grid-stride loop – each thread handles several output elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        // ---- decode flat index to (n, oc, od, oh, ow) ----
        int n = i / (C_out * D_out * H_out * W_out);
        int rem = i % (C_out * D_out * H_out * W_out);
        int oc = rem / (D_out * H_out * W_out);
        rem   = rem % (D_out * H_out * W_out);
        int od = rem / (H_out * W_out);
        rem   = rem % (H_out * W_out);
        int oh = rem / W_out;
        int ow = rem % W_out;

        float sum = 0.0f;
        if (bias) sum += bias[oc];

        // ---- transposed convolution (group = 1) ----
        for (int ic = 0; ic < C_in; ++ic) {
            int weight_base = (oc * C_in + ic) * kernel_vol;
            // loop over kernel dimensions – the compiler will unroll these
            for (int kd = 0; kd < kernel_size; ++kd) {
                int pad_d = od + kd;
                int up_d = pad_d - padding;
                if (up_d < 0) continue;
                int id = up_d / stride;
                if (up_d % stride != 0) continue;
                if (id >= D_in) continue;

                for (int kh = 0; kh < kernel_size; ++kh) {
                    int pad_h = oh + kh;
                    int up_h = pad_h - padding;
                    if (up_h < 0) continue;
                    int ih = up_h / stride;
                    if (up_h % stride != 0) continue;
                    if (ih >= H_in) continue;

                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int pad_w = ow + kw;
                        int up_w = pad_w - padding;
                        if (up_w < 0) continue;
                        int iw = up_w / stride;
                        if (up_w % stride != 0) continue;
                        if (iw >= W_in) continue;

                        int kidx = ((kd * kernel_size + kh) * kernel_size + kw);
                        float w = __ldg(&weight[weight_base + kidx]);

                        // input is stored as (N, C_in, D, H, W) – flatten to 1-D
                        int inp_idx = ((n * C_in + ic) * D_in + id) * (H_in * W_in)
                                    + ih * W_in + iw;
                        float inp = __ldg(&input[inp_idx]);

                        sum += inp * w;
                    }
                }
            }
        }

        // ---- addition of the "add_input" tensor ----
        int add_idx = ((n * C_out + oc) * D_out + od) * (H_out * W_out)
                    + oh * W_out + ow;
        sum += __ldg(&add_input[add_idx]);

        // ---- hardswish activation: x * ReLU6(x+3) / 6 ----
        float tmp   = sum + 3.0f;
        float relu6 = fminf(fmaxf(tmp, 0.0f), 6.0f);
        float result = sum * relu6 / 6.0f;

        // ---- write final output ----
        output[add_idx] = result;
    }
}

void fused_conv_transpose_add_hardswish(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& add_input,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int kernel_size)
{
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    const int C_out = weight.size(0);
    const int D_out = output.size(2);
    const int H_out = output.size(3);
    const int W_out = output.size(4);

    const int block = 256;
    const int total = N * C_out * D_out * H_out * W_out;
    const int grid  = (total + block - 1) / block;

    fused_conv_transpose_add_hardswish_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        add_input.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        stride, padding, kernel_size);
}
"""

# ----------------------------------------------------------------------
# C++ bindings (PYBIND11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_add_hardswish(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& add_input,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int kernel_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_add_hardswish",
          &fused_conv_transpose_add_hardswish,
          "Fused conv_transpose3d + element-wise add + hardswish");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
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
    add_input,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    bias=None,          # dummy argument – not used
):
    # Make sure all inputs are contiguous CUDA tensors
    x = x.contiguous().cuda()
    add_input = add_input.contiguous().cuda()
    conv_transpose_weight = conv_transpose_weight.contiguous().cuda()

    # If no bias is supplied we pass a zero tensor – the kernel will simply ignore it
    if conv_transpose_bias is None:
        conv_transpose_bias = torch.zeros(
            conv_transpose_weight.size(0), dtype=torch.float32, device='cuda'
        )
    else:
        conv_transpose_bias = conv_transpose_bias.contiguous().cuda()

    # ------------------------------------------------------------------
    # Compute output spatial size using the official transposed-conv formula
    # ------------------------------------------------------------------
    D_in = x.size(2)
    H_in = x.size(3)
    W_in = x.size(4)

    stride = conv_transpose_stride[0] if isinstance(conv_transpose_stride, (list, tuple)) else conv_transpose_stride
    padding = conv_transpose_padding[0] if isinstance(conv_transpose_padding, (list, tuple)) else conv_transpose_padding
    output_padding = conv_transpose_output_padding
    dilation = conv_transpose_dilation[0] if isinstance(conv_transpose_dilation, (list, tuple)) else conv_transpose_dilation
    kernel_size = conv_transpose_weight.size(2)   # square kernel is assumed

    D_out = (D_in - 1) * stride - 2 * padding + output_padding + \
            dilation * (kernel_size - 1) + 1
    H_out = (H_in - 1) * stride - 2 * padding + output_padding + \
            dilation * (kernel_size - 1) + 1
    W_out = (W_in - 1) * stride - 2 * padding + output_padding + \
            dilation * (kernel_size - 1) + 1

    batch = x.size(0)
    out_channels = conv_transpose_weight.size(0)

    # Allocate output tensor on the GPU
    output = torch.empty(
        (batch, out_channels, D_out, H_out, W_out),
        dtype=torch.float32,
        device='cuda'
    )

    # ------------------------------------------------------------------
    # Launch the fused CUDA kernel
    # ------------------------------------------------------------------
    fused_ext.fused_conv_transpose_add_hardswish(
        x,
        conv_transpose_weight,
        add_input,
        conv_transpose_bias,
        output,
        stride,
        padding,
        kernel_size
    )

    return output
