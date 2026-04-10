# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_152515/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'add_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'add_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

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
    if 'add_value' in flat_state:
        state_kwargs['add_value'] = flat_state['add_value']
    else:
        state_kwargs['add_value'] = getattr(model, 'add_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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

# ------------------------------------------------------------
# 1. CUDA source – fused transposed‑convolution + pointwise ops
# ------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float gelu(float x)
{
    // Fast GELU approximation:
    //   x * 0.5 * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float c = 0.044715f;
    float x3 = x * x * x;
    float t = sqrt_2_over_pi * (x + c * x3);
    return x * 0.5f * (1.0f + tanhf(t));
}

__global__ void fused_deconv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int H_in, const int W_in,
    const int H_out, const int W_out,
    const int K, const int stride, const int padding,
    const int output_padding, const int dilation,
    const float add_val,
    const float mul_val)
{
    // ---- locate the output element this thread is responsible for ----
    const int total_out = N * C_out * H_out * W_out;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_out) return;

    // unflatten linear index → (n, oc, oh, ow)
    int n   = tid / (C_out * H_out * W_out);
    int rem = tid % (C_out * H_out * W_out);
    int oc  = rem / (H_out * W_out);
    rem    %= (H_out * W_out);
    int oh  = rem / W_out;
    int ow  = rem % W_out;

    // ----- transposed convolution accumulation -----
    float acc = 0.0f;
    if (bias) acc += bias[oc];

    for (int ic = 0; ic < C_in; ++ic)
    {
        for (int kh = 0; kh < K; ++kh)
        {
            int ih = (oh + padding - kh * dilation);
            if (ih % stride != 0) continue;
            ih /= stride;
            if (ih < 0 || ih >= H_in) continue;

            for (int kw = 0; kw < K; ++kw)
            {
                int iw = (ow + padding - kw * dilation);
                if (iw % stride != 0) continue;
                iw /= stride;
                if (iw < 0 || iw >= W_in) continue;

                // weight index assuming layout (C_in, C_out, K, K)
                int w_idx = ((ic * C_out + oc) * K + kh) * K + kw;
                float w = __ldg(&weight[w_idx]);

                // input index (NCHW)
                int in_idx = ((n * C_in + ic) * H_in + ih) * W_in + iw;
                float v = __ldg(&input[in_idx]);

                acc += v * w;
            }
        }
    }

    // ----- point‑wise post‑ops (add → clamp → gelu → mul) -----
    float x = acc + add_val;
    x = fminf(x, 0.0f);          // min(x,0)
    x = gelu(x);
    x = x * mul_val;

    // store result
    int out_idx = ((n * C_out + oc) * H_out + oh) * W_out + ow;
    output[out_idx] = x;
}

// Host wrapper called from Python
void fused_op(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor output,
    int N, int C_in, int C_out,
    int H_in, int W_in,
    int H_out, int W_out,
    int K, int stride, int padding,
    int output_padding, int dilation,
    float add_val,
    float mul_val)
{
    const int total = N * C_out * H_out * W_out;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;

    fused_deconv_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, C_out,
        H_in, W_in,
        H_out, W_out,
        K, stride, padding,
        output_padding, dilation,
        add_val, mul_val);
    cudaDeviceSynchronize();
}
"""

# ------------------------------------------------------------
# 2. C++/pybind11 binding – exposes fused_op to Python
# ------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor output,
    int N, int C_in, int C_out,
    int H_in, int W_in,
    int H_out, int W_out,
    int K, int stride, int padding,
    int output_padding, int dilation,
    float add_val,
    float mul_val);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op,
          "Fused transposed convolution + element‑wise ops");
}
"""

# ------------------------------------------------------------
# 3. Build the inline CUDA extension
# ------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ------------------------------------------------------------
# 4. Replace the original functional_model with the fused version
# ------------------------------------------------------------
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
    add_value,
    multiply_value,
):
    """
    Fused transposed convolution + add + clamp(·,0) + GELU + multiply.
    All arithmetic is performed inside a single CUDA kernel, eliminating
    the kernel‑launch overhead of the original pipeline.
    """
    # --------------------------------------------------------
    # Basic shape information
    # --------------------------------------------------------
    N, C_in, H_in, W_in = x.shape           # input (N, C_in, H_in, W_in)
    K = conv_transpose_weight.shape[2]       # square kernel (4 in the benchmark)
    C_out = conv_transpose_weight.shape[1]   # output channels

    # --------------------------------------------------------
    # Ensure weight layout is (C_in, C_out, K, K)
    # Some frameworks store it as (C_out, C_in, K, K); transpose if needed.
    # --------------------------------------------------------
    if conv_transpose_weight.shape[0] == C_out and conv_transpose_weight.shape[1] == C_in:
        weight = conv_transpose_weight.transpose(0, 1).contiguous()
    else:
        weight = conv_transpose_weight.contiguous()

    # --------------------------------------------------------
    # Bias: use a zero tensor when not supplied (keeps the kernel simple)
    # --------------------------------------------------------
    if conv_transpose_bias is None:
        bias = torch.zeros(C_out, dtype=x.dtype, device=x.device)
    else:
        bias = conv_transpose_bias.contiguous()

    # --------------------------------------------------------
    # Compute output spatial size using the standard transposed‑conv formula
    # --------------------------------------------------------
    H_out = (H_in - 1) * conv_transpose_stride \
            - 2 * conv_transpose_padding \
            + conv_transpose_dilation * (K - 1) \
            + conv_transpose_output_padding + 1
    W_out = (W_in - 1) * conv_transpose_stride \
            - 2 * conv_transpose_padding \
            + conv_transpose_dilation * (K - 1) \
            + conv_transpose_output_padding + 1

    # --------------------------------------------------------
    # Allocate output tensor
    # --------------------------------------------------------
    out = torch.empty((N, C_out, H_out, W_out),
                      dtype=x.dtype, device=x.device)

    # --------------------------------------------------------
    # Launch the fused CUDA kernel
    # --------------------------------------------------------
    fused_ext.fused_op(
        x.contiguous(),
        weight,
        bias,
        out,
        N, C_in, C_out,
        H_in, W_in,
        H_out, W_out,
        K,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_dilation,
        add_value,
        multiply_value
    )
    return out


# --------------------------------------------------------------------
# The remaining code is only for illustration / quick sanity checks.
# --------------------------------------------------------------------
batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


if __name__ == "__main__":
    # ---- simple correctness test (not required by the evaluation) ----
    import torch.nn.functional as F

    w = torch.randn(in_channels, out_channels, kernel_size, kernel_size)
    b = torch.randn(out_channels) if torch.rand(1).item() > 0.5 else None
    x = torch.rand(batch_size, in_channels, height, width)

    # reference result (original pipeline)
    out_ref = F.conv_transpose2d(x, w, b, stride=stride, padding=1,
                                 output_padding=0, groups=1, dilation=1)
    out_ref = out_ref + add_value
    out_ref = torch.min(out_ref, torch.tensor(0.0, device=out_ref.device))
    out_ref = F.gelu(out_ref)
    out_ref = out_ref * multiply_value

    # fused result
    out_fused = functional_model(
        x,
        conv_transpose_weight=w,
        conv_transpose_bias=b,
        conv_transpose_stride=stride,
        conv_transpose_padding=1,
        conv_transpose_output_padding=0,
        conv_transpose_groups=1,
        conv_transpose_dilation=1,
        add_value=add_value,
        multiply_value=multiply_value,
    )

    print("Maximum absolute difference:", (out_ref - out_fused).abs().max().item())
