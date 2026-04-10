# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_145317/code_8.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'subtract_value_1', 'subtract_value_2']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'subtract_value_1', 'subtract_value_2']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, subtracts two values, applies Mish activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

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
    if 'subtract_value_1' in flat_state:
        state_kwargs['subtract_value_1'] = flat_state['subtract_value_1']
    else:
        state_kwargs['subtract_value_1'] = getattr(model, 'subtract_value_1')
    if 'subtract_value_2' in flat_state:
        state_kwargs['subtract_value_2'] = flat_state['subtract_value_2']
    else:
        state_kwargs['subtract_value_2'] = getattr(model, 'subtract_value_2')
    missing = [name for name in REQUIRED_STATE_NAMES if name not in state_kwargs]
    if missing:
        raise RuntimeError(f'Missing required state entries: {missing}')
    return state_kwargs


def get_functional_inputs():
    model = build_reference_model()
    forward_args = tuple(get_inputs())
    state_kwargs = extract_state_kwargs(model)
    return forward_args, state_kwargs




# -*- coding: utf-8 -*-
"""
Optimised fused convolution + Mish activation for RTX 2080 Ti.
The only optimisation applied is a grid‑stride loop over the output
spatial domain.
"""

import torch
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------------------------
# CUDA kernel – grid‑stride version
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

#define TILE 16          // threads per dimension inside a block
#define WARP_SIZE 32

// ---------------------------------------------------------------
// Mish(x) = x * tanh(softplus(x))  ; softplus(x)=log1p(exp(x))
// ---------------------------------------------------------------
static __device__ __forceinline__ float mish(float x) {
    // fast approximations are already enabled via --use_fast_math
    float sp = logf(1.0f + expf(x));
    return x * tanhf(sp);
}

// ---------------------------------------------------------------
// One block processes a single (batch, out_channel) pair.
// Inside the block every thread walks over many output pixels
// using a grid‑stride loop.
// ---------------------------------------------------------------
extern "C"
__global__ void fused_conv_mish_kernel(
        const float* __restrict__ input,   // [B, Cin, Hin, Win]
        const float* __restrict__ weight,  // [Cout, K*K*Cin]   (reordered)
        const float* __restrict__ bias,    // [Cout]
        float* __restrict__ output,        // [B, Cout, Hout, Wout]
        int batch,
        int in_c, int in_h, int in_w,
        int out_c,
        int k,
        int out_h, int out_w,
        float sub1,
        float sub2)
{
    // -----------------------------------------------------------------
    // Block identifies which (batch, out_channel) it works on
    // -----------------------------------------------------------------
    const int oc = blockIdx.y;          // output channel
    const int b  = blockIdx.z;          // batch index

    // -----------------------------------------------------------------
    // 1) Load kernel weights for this output channel into shared memory
    // -----------------------------------------------------------------
    extern __shared__ float weight_smem[];   // size = K*K*Cin
    const int weight_vol = k * k * in_c;

    // Collaborative loading – each thread copies several elements
    for (int i = threadIdx.y * blockDim.x + threadIdx.x;
         i < weight_vol;
         i += blockDim.x * blockDim.y) {
        weight_smem[i] = weight[oc * weight_vol + i];
    }
    __syncthreads();

    // -----------------------------------------------------------------
    // 2) Grid‑stride loop over the *flattened* output plane
    // -----------------------------------------------------------------
    const int out_plane = out_h * out_w;                     // per (b,oc)
    const int lane_id   = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_sz  = blockDim.x * blockDim.y;
    const int grid_sz   = gridDim.x * block_sz;              // total threads in x‑dimension

    // start index for this thread
    int idx = blockIdx.x * block_sz + lane_id;

    const float bias_val = bias[oc];

    while (idx < out_plane) {
        // -------------------------------------------------------------
        // Convert linear index back to (oh, ow)
        // -------------------------------------------------------------
        const int oh = idx / out_w;
        const int ow = idx % out_w;

        // -------------------------------------------------------------
        // Compute convolution for this output pixel
        // -------------------------------------------------------------
        float acc = bias_val;

        // Walk over the kernel window
        #pragma unroll
        for (int ki = 0; ki < k; ++ki) {
            const int ih = oh + ki;                     // input row
            const int base_w_ki = ki * k * in_c;        // weight offset for row ki

            #pragma unroll
            for (int kj = 0; kj < k; ++kj) {
                const int iw = ow + kj;                 // input column
                const int base_w = base_w_ki + kj * in_c; // weight offset for (ki,kj)

                // Base address of the current input column slice
                const int in_base = (b * in_c * in_h * in_w) + (ih * in_w + iw);

                // Accumulate over input channels
                #pragma unroll
                for (int ic = 0; ic < in_c; ++ic) {
                    // input layout: [B, Cin, Hin, Win]  => contiguous over W then H then C then B
                    float val = input[in_base + ic * (in_h * in_w)];
                    acc += val * weight_smem[base_w + ic];
                }
            }
        }

        // -------------------------------------------------------------
        // Mish activation with the two subtraction constants
        // -------------------------------------------------------------
        float v = acc - sub1 - sub2;
        output[((b * out_c + oc) * out_h + oh) * out_w + ow] = mish(v);

        // Move to next element handled by this thread
        idx += grid_sz;
    }
}

// ---------------------------------------------------------------------
// C++ wrapper called from Python
// ---------------------------------------------------------------------
void fused_conv_mish(
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor bias,
        torch::Tensor output,
        float sub1,
        float sub2)
{
    const int batch = input.size(0);
    const int in_c   = input.size(1);
    const int in_h   = input.size(2);
    const int in_w   = input.size(3);
    const int out_c  = weight.size(0);
    const int k      = weight.size(1);            // kernel size (assumed square)
    const int out_h  = in_h - k + 1;
    const int out_w  = in_w - k + 1;

    dim3 threads(TILE, TILE);
    // one block per (out_h*out_w) chunk of TILE*TILE pixels
    const int out_plane = out_h * out_w;
    const int blocks_x = (out_plane + TILE*TILE - 1) / (TILE*TILE);
    dim3 blocks(blocks_x, out_c, batch);

    size_t shared_mem_bytes = static_cast<size_t>(k) * k * in_c * sizeof(float);

    fused_conv_mish_kernel<<<blocks, threads, shared_mem_bytes>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, in_c, in_h, in_w,
        out_c, k, out_h, out_w,
        sub1, sub2);

    // Propagate any launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration
void fused_conv_mish(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    float sub1,
    float sub2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &fused_conv_mish, "Fused Conv + Mish (grid‑stride version)");
}
"""

# -------------------------------------------------------------------------
# Build the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name="fused_ext_grid_stride",
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)

# -------------------------------------------------------------------------
# Public API – functional_model (the only symbol used by the evaluator)
# -------------------------------------------------------------------------
def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride=1,
    conv_padding=0,
    conv_dilation=1,
    conv_groups=1,
    subtract_value_1,
    subtract_value_2,
):
    """
    Performs a valid (no‑pad, stride‑1) convolution followed by
    Mish activation where the output of the activation is additionally
    shifted by `subtract_value_1` and `subtract_value_2`.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, Cin, H, W) – must be contiguous.
    conv_weight : torch.Tensor
        Convolution weight of shape (Cout, Cin, K, K).
    conv_bias : torch.Tensor
        Bias tensor of shape (Cout,).
    subtract_value_1, subtract_value_2 : float
        Scalars subtracted before the Mish non‑linearity.

    Returns
    -------
    torch.Tensor
        Tensor of shape (B, Cout, H‑K+1, W‑K+1) on the same device as `x`.
    """
    # ---- Re‑order weight to [Cout, K, K, Cin] as expected by the kernel ----
    # Original layout: (Cout, Cin, K, K)
    w_reordered = conv_weight.permute(0, 2, 3, 1).contiguous()

    B, _, H, W = x.shape
    K = conv_weight.shape[2]               # kernel size (square)
    out_h = H - K + 1
    out_w = W - K + 1

    out = torch.empty((B, conv_weight.size(0), out_h, out_w), device=x.device, dtype=x.dtype)

    # Call the custom fused kernel
    fused_ext.fused_conv(
        x,
        w_reordered,
        conv_bias,
        out,
        float(subtract_value_1),
        float(subtract_value_2),
    )
    return out
