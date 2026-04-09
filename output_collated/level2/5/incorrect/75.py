# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_123254/code_13.py
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
# CUDA source – fused transposed‑convolution + both biases + tanh
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_conv_bias_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,      // bias added inside the transposed convolution
    const float* __restrict__ final_bias,     // bias to subtract before tanh
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int H_in, const int W_in,
    const int K_h, const int K_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int out_pad_h, const int out_pad_w,
    const int dil_h, const int dil_w,
    const int groups,
    const int H_out_base, const int W_out_base,
    const int H_out, const int W_out
) {
    // Shared memory for both bias vectors (conv_bias + final_bias)
    extern __shared__ float shared_bias[];
    const int C = C_out;
    const int tid = threadIdx.x;

    // Cooperative loading of bias vectors
    if (tid < C) {
        shared_bias[tid]                = conv_bias[tid];
        shared_bias[tid + C]            = final_bias[tid];
    }
    __syncthreads();

    const float* conv_bias_shared = shared_bias;
    const float* final_bias_shared = shared_bias + C;

    // Compute global linear index of the output element
    const int total_out = N * C_out * H_out * W_out;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    // Decode (n, oc, oh, ow)
    int remainder = idx;
    const int n = remainder / (C_out * H_out * W_out);
    remainder %= (C_out * H_out * W_out);
    const int oc = remainder / (H_out * W_out);
    remainder %= (H_out * W_out);
    const int oh = remainder / W_out;
    const int ow = remainder % W_out;

    // If we are in the output‑padding region (outside the base computed size) just zero the sum
    float sum = 0.0f;
    if (oh < H_out_base && ow < W_out_base) {
        // Determine group information
        const int group_size_out = C_out / groups;
        const int group_size_in  = C_in  / groups;
        const int gid = oc / group_size_out;                // which group we belong to
        const int ocg = oc % group_size_out;                // output channel inside the group

        // Loop over input channels of this group
        for (int ic = 0; ic < group_size_in; ++ic) {
            const int ic_full = gid * group_size_in + ic;   // absolute input channel index

            // Base pointer for the weight sub‑tensor of this group
            const int weight_group_offset = gid * group_size_in * group_size_out;
            const int weight_ic_offset = (weight_group_offset + ic * group_size_out + ocg) * (K_h * K_w);

            // Kernel loops
            for (int kh = 0; kh < K_h; ++kh) {
                // Input row candidate
                int ih = (oh + pad_h - kh * dil_h);
                if (ih % stride_h != 0) continue;
                ih /= stride_h;
                if (ih < 0 || ih >= H_in) continue;

                for (int kw = 0; kw < K_w; ++kw) {
                    // Input column candidate
                    int iw = (ow + pad_w - kw * dil_w);
                    if (iw % stride_w != 0) continue;
                    iw /= stride_w;
                    if (iw < 0 || iw >= W_in) continue;

                    // Input value (coalesced read)
                    const int input_idx = ((n * C_in + ic_full) * H_in + ih) * W_in + iw;
                    const float in_val = input[input_idx];

                    // Weight value
                    const int weight_idx = weight_ic_offset + (kh * K_w + kw);
                    const float w_val = weight[weight_idx];

                    sum += in_val * w_val;
                }
            }
        }
    }

    // Add the convolution bias (stored in shared memory)
    sum += conv_bias_shared[oc];

    // Subtract the user‑provided bias and apply tanh
    sum = tanhf(sum - final_bias_shared[oc]);

    // Write result (coalesced write)
    output[idx] = sum;
}

void fused_conv_bias_tanh(
    torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& final_bias,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int out_pad_h, const int out_pad_w,
    const int dil_h, const int dil_w,
    const int groups,
    const int H_out_base, const int W_out_base,
    const int H_out, const int W_out,
    torch::Tensor& output
) {
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int C_out = weight.size(1) * groups;
    const int total_out = N * C_out * H_out * W_out;

    const int threads = 256;
    const int blocks = (total_out + threads - 1) / threads;

    // Two bias vectors stored consecutively in shared memory
    const size_t shared_mem = 2 * C_out * sizeof(float);

    fused_conv_bias_tanh_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        final_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out,
        input.size(2), input.size(3),
        weight.size(2), weight.size(3),
        stride_h, stride_w,
        pad_h, pad_w,
        out_pad_h, out_pad_w,
        dil_h, dil_w,
        groups,
        H_out_base, W_out_base,
        H_out, W_out
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_bias_tanh(
    torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& final_bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_pad_h, int out_pad_w,
    int dil_h, int dil_w,
    int groups,
    int H_out_base, int W_out_base,
    int H_out, int W_out,
    torch::Tensor& output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_bias_tanh", &fused_conv_bias_tanh,
          "Fused transposed convolution, bias addition, bias subtraction and tanh activation");
}
"""

# Compile the inline extension
fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional model – replaces the original two‑stage pipeline
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
    # Unpack convolution parameters (assume they are either ints or 2‑element tuples)
    # -----------------------------------------------------------------
    stride_h = conv_transpose_stride[0] if isinstance(conv_transpose_stride, (tuple, list)) else conv_transpose_stride
    stride_w = conv_transpose_stride[1] if isinstance(conv_transpose_stride, (tuple, list)) else stride_h

    pad_h = conv_transpose_padding[0] if isinstance(conv_transpose_padding, (tuple, list)) else conv_transpose_padding
    pad_w = conv_transpose_padding[1] if isinstance(conv_transpose_padding, (tuple, list)) else pad_h

    out_pad_h = conv_transpose_output_padding[0] if isinstance(conv_transpose_output_padding, (tuple, list)) else 0
    out_pad_w = conv_transpose_output_padding[1] if isinstance(conv_transpose_output_padding, (tuple, list)) else out_pad_h

    dil_h = conv_transpose_dilation[0] if isinstance(conv_transpose_dilation, (tuple, list)) else conv_transpose_dilation
    dil_w = conv_transpose_dilation[1] if isinstance(conv_transpose_dilation, (tuple, list)) else dil_h

    groups = conv_transpose_groups

    # -----------------------------------------------------------------
    # Input tensor shape
    # -----------------------------------------------------------------
    N = x.size(0)
    C_in = x.size(1)
    H_in = x.size(2)
    W_in = x.size(3)

    # -----------------------------------------------------------------
    # Weight shape: (C_in, C_out/groups, K_h, K_w)
    # -----------------------------------------------------------------
    K_h = conv_transpose_weight.size(2)
    K_w = conv_transpose_weight.size(3)
    C_out = conv_transpose_weight.size(1) * groups   # total number of output channels

    # -----------------------------------------------------------------
    # Output size without output padding (the “base” size)
    # -----------------------------------------------------------------
    H_out_base = (H_in - 1) * stride_h - 2 * pad_h + dil_h * (K_h - 1) + 1
    W_out_base = (W_in - 1) * stride_w - 2 * pad_w + dil_w * (K_w - 1) + 1

    # Add output padding to obtain the final output shape
    H_out = H_out_base + out_pad_h
    W_out = W_out_base + out_pad_w

    # -----------------------------------------------------------------
    # Allocate output tensor
    # -----------------------------------------------------------------
    output = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)

    # -----------------------------------------------------------------
    # Prepare bias tensors (flatten to 1‑D)
    # -----------------------------------------------------------------
    # Convolution‑inside bias (may be None in which case we use zeros)
    if conv_transpose_bias is None:
        conv_bias = torch.zeros(C_out, dtype=x.dtype, device=x.device)
    else:
        conv_bias = conv_transpose_bias.view(-1).contiguous()

    # Final bias (to be subtracted before tanh)
    final_bias = bias.view(-1).contiguous() if bias is not None else torch.zeros(C_out, dtype=x.dtype, device=x.device)

    # -----------------------------------------------------------------
    # Ensure weight is contiguous (PyTorch default is row‑major, which matches our indexing)
    # -----------------------------------------------------------------
    weight = conv_transpose_weight.contiguous()

    # -----------------------------------------------------------------
    # Launch the fused kernel
    # -----------------------------------------------------------------
    fused_ext.fused_conv_bias_tanh(
        x,                      # input
        weight,                 # conv_transpose weight
        conv_bias,              # bias added inside the transposed convolution
        final_bias,             # bias to subtract before tanh
        stride_h, stride_w,
        pad_h, pad_w,
        out_pad_h, out_pad_w,
        dil_h, dil_w,
        groups,
        H_out_base, W_out_base,
        H_out, W_out,
        output
    )

    return output
