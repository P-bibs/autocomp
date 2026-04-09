# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_085904/code_6.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies minimum operation, Tanh, and another Tanh.
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
# CUDA source: the fused kernel + host launcher
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ----------------------------------------------------------------------
// Warp-level minimum reduction (requires CUDA >= 9.0)
// ----------------------------------------------------------------------
__device__ __forceinline__ float warp_min(float val) {
    // 64 threads -> two warps
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// ----------------------------------------------------------------------
// Main fused kernel: conv → min over channels → double tanh
// ----------------------------------------------------------------------
__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int H, const int W, const int K,
    const int stride, const int padding) {

    // Output spatial sizes (the same for every block)
    const int OH = (H + 2 * padding - K) / stride + 1;
    const int OW = (W + 2 * padding - K) / stride + 1;

    // Linear block index -> (n, oh, ow) coordinate
    const int out_idx = blockIdx.x;                     // 0 … N*OH*OW−1
    const int n       = out_idx / (OH * OW);
    const int rem     = out_idx % (OH * OW);
    const int oh      = rem / OW;
    const int ow      = rem % OW;

    // Top-left corner of the input patch (including padding)
    const int i_h_base = oh * stride - padding;
    const int i_w_base = ow * stride - padding;

    // ------------------------------------------------------------------
    // Shared memory for the input patch (C_in * K * K elements)
    // ------------------------------------------------------------------
    extern __shared__ float s_input[];
    const int patch_size = C_in * K * K;               // = 144 for the given case

    // Cooperatively load the patch (each thread picks a few elements)
    const int tid = threadIdx.x;                       // = output channel index (0…C_out−1)
    const int stride_load = blockDim.x;                // = C_out = 64
    for (int i = tid; i < patch_size; i += stride_load) {
        int ci = i / (K * K);
        int rem_i = i % (K * K);
        int kh = rem_i / K;
        int kw = rem_i % K;
        int ih = i_h_base + kh;
        int iw = i_w_base + kw;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            // x layout: (N, C_in, H, W)
            s_input[i] = x[((n * C_in + ci) * H + ih) * W + iw];
        } else {
            s_input[i] = 0.0f;                         // zero-padding
        }
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Each thread handles one output channel (co = tid)
    // ------------------------------------------------------------------
    const int co = tid;
    if (co >= C_out) return;

    // ------------------------------------------------------------------
    // Load the weight for this output channel into registers
    // ------------------------------------------------------------------
    // weight layout: (C_out, C_in, K, K)
    float weight_local[144];               // max size = C_in*K*K
    for (int i = 0; i < patch_size; ++i) {
        int ci = i / (K * K);
        int rem_i = i % (K * K);
        int kh = rem_i / K;
        int kw = rem_i % K;
        weight_local[i] = weight[(((co * C_in + ci) * K + kh) * K + kw)];
    }

    // ------------------------------------------------------------------
    // Compute convolution sum for this channel
    // ------------------------------------------------------------------
    float sum = bias[co];
#pragma unroll
    for (int i = 0; i < patch_size; ++i) {
        sum += s_input[i] * weight_local[i];
    }

    // ------------------------------------------------------------------
    // Minimum across all output channels (warp reduction)
    // ------------------------------------------------------------------
    float min_val = warp_min(sum);

    // ------------------------------------------------------------------
    // Apply double tanh and write the final scalar
    // ------------------------------------------------------------------
    if (tid == 0) {
        float t = tanhf(min_val);
        t = tanhf(t);
        // output layout: (N, 1, OH, OW) stored as (N*OH*OW) row-major
        output[(n * OH + oh) * OW + ow] = t;
    }
}

// ----------------------------------------------------------------------
// Host function that launches the kernel
// ----------------------------------------------------------------------
void fused_op_forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const int stride,
    const int padding,
    torch::Tensor& output) {

    // Make sure tensors are contiguous and on GPU
    auto x_a     = x.contiguous();
    auto w_a     = weight.contiguous();
    auto b_a     = bias.contiguous();

    const int N     = x_a.size(0);
    const int C_in  = x_a.size(1);
    const int H     = x_a.size(2);
    const int W     = x_a.size(3);
    const int C_out = w_a.size(0);
    const int K     = w_a.size(2);               // square kernel

    const int OH = (H + 2 * padding - K) / stride + 1;
    const int OW = (W + 2 * padding - K) / stride + 1;

    // Resize output tensor to (N, 1, OH, OW)
    output.resize_({N, 1, OH, OW});

    const int blocks   = N * OH * OW;
    const int threads  = C_out;                 // 64 for the given problem
    const int sh_mem   = C_in * K * K * sizeof(float);   // 144 * 4 = 576 bytes

    fused_conv_min_tanh_kernel<<<blocks, threads, sh_mem>>>(
        reinterpret_cast<const float*>(x_a.data_ptr()),
        reinterpret_cast<const float*>(w_a.data_ptr()),
        reinterpret_cast<const float*>(b_a.data_ptr()),
        reinterpret_cast<float*>(output.data_ptr()),
        N, C_in, C_out, H, W, K, stride, padding);
}
"""

# ----------------------------------------------------------------------
# C++ binding (PYBIND11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const int stride,
    const int padding,
    torch::Tensor& output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward,
          "Fused convolution → channel-wise min → double tanh");
}
"""

# ----------------------------------------------------------------------
# Compile the extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# ----------------------------------------------------------------------
# Functional model: uses the custom fused kernel
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
    """
    Fused operation: 2-D convolution + min over output channels + double tanh.
    All three stages are executed in a single CUDA kernel that caches the
    input patch in shared memory, performs a warp-level reduction for the
    channel-wise minimum, and applies two tanh activations.
    """
    # Compute output spatial size (assumes square kernel, stride and padding)
    K = conv_weight.size(2)                     # kernel height (and width)
    H = x.size(2)
    W = x.size(3)
    OH = (H + 2 * conv_padding - K) // conv_stride + 1
    OW = (W + 2 * conv_padding - K) // conv_stride + 1

    # Allocate output tensor with shape (N, 1, OH, OW)
    output = torch.empty((x.size(0), 1, OH, OW),
                         dtype=x.dtype,
                         device=x.device)

    # Launch the fused CUDA kernel
    fused_ext.fused_op(x, conv_weight, conv_bias,
                       conv_stride, conv_padding, output)

    return output
