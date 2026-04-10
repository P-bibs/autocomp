# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_150347/code_13.py
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




import torch
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# Optimized CUDA kernel – loop invariants hoisted
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Dynamic shared memory for the weight tile of one output channel
extern __shared__ float weight_cache[];

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
    const int in_c,
    const int in_h,
    const int in_w,
    const int out_c,
    const int k,
    const float sub1,
    const float sub2)
{
    const int out_h = in_h - k + 1;
    const int out_w = in_w - k + 1;

    const int oc = blockIdx.y;            // output channel for this block
    const int b  = blockIdx.z;             // batch index

    const int tid = threadIdx.x;
    const int block_threads = blockDim.x;

    // ---- Load weight tile for this output channel into shared memory ----
    const int weight_tile_sz = k * k * in_c;
    const int weight_offset = oc * weight_tile_sz;
    for (int idx = tid; idx < weight_tile_sz; idx += block_threads) {
        weight_cache[idx] = __ldg(&weight[weight_offset + idx]);
    }
    __syncthreads();

    // ---- Compute spatial index for this thread ----
    const int num_spatial = out_h * out_w;
    const int spatial_idx = blockIdx.x * block_threads + tid;
    if (spatial_idx >= num_spatial) return;

    const int oh = spatial_idx / out_w;
    const int ow = spatial_idx % out_w;

    // ---- Convolution accumulation (loop‑invariant calculations hoisted) ----
    float acc = __ldg(&bias[oc]);          // load bias once

    const float* i_base = input + (b * in_c * in_h * in_w);

    // Loop over input channels – hoisted pointer per channel
    for (int ic = 0; ic < in_c; ++ic) {
        const float* i_channel = i_base + ic * in_h * in_w;   // base pointer per channel

        // Kernel rows – row offset computed once per row
        for (int i = 0; i < k; ++i) {
            const int ih        = oh + i;
            const int row_base  = ih * in_w + ow;             // once per row

            // Kernel columns – only an addition inside the innermost loop
            for (int j = 0; j < k; ++j) {
                const int offset      = row_base + j;
                const int weight_base = (i * k + j) * in_c;   // base index per (i,j)

                const float inp = __ldg(&i_channel[offset]); // coalesced read
                const float w   = weight_cache[weight_base + ic];
                acc += inp * w;
            }
        }
    }

    // ---- Mish activation after subtraction ----
    const float val = acc - sub1 - sub2;
    // mish = x * tanh(ln(1 + e^x))
    const float mish = val * tanhf(logf(1.0f + expf(val)));

    // ---- Store result ----
    const int out_idx = ((b * out_c + oc) * out_h + oh) * out_w + ow;
    output[out_idx] = mish;
}

// Host wrapper that launches the kernel with dynamic shared memory
void fused_conv_mish(torch::Tensor input,
                     torch::Tensor weight,
                     torch::Tensor bias,
                     torch::Tensor output,
                     float sub1,
                     float sub2)
{
    const int batch   = input.size(0);
    const int in_c    = input.size(1);
    const int in_h    = input.size(2);
    const int in_w    = input.size(3);
    const int out_c   = weight.size(0);
    const int k       = weight.size(2);
    const int out_h   = in_h - k + 1;
    const int out_w   = in_w - k + 1;

    const int numSpatial = out_h * out_w;
    const int threads = 256;                     // 1‑D block, multiple of 32
    const int blocksSpatial = (numSpatial + threads - 1) / threads;

    dim3 grid(blocksSpatial, out_c, batch);
    dim3 block(threads);                         // 256 threads per block

    // Dynamic shared memory = size of weight tile for one output channel
    const size_t shared_mem = k * k * in_c * sizeof(float);

    fused_conv_mish_kernel<<<grid, block, shared_mem>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, in_c, in_h, in_w,
        out_c, k, sub1, sub2);
}
"""

# ----------------------------------------------------------------------
# C++ binding (PyBind11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_mish(torch::Tensor input,
                     torch::Tensor weight,
                     torch::Tensor bias,
                     torch::Tensor output,
                     float sub1,
                     float sub2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &fused_conv_mish,
          "Fused convolution followed by Mish activation");
}
"""

# ----------------------------------------------------------------------
# Compile the inline CUDA extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_conv_mish',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional model – entry point used for evaluation
# ----------------------------------------------------------------------
def functional_model(x: torch.Tensor,
                     *,
                     conv_weight: torch.Tensor,
                     conv_bias: torch.Tensor,
                     conv_stride: int = 1,
                     conv_padding: int = 0,
                     conv_dilation: int = 1,
                     conv_groups: int = 1,
                     subtract_value_1: float,
                     subtract_value_2: float) -> torch.Tensor:
    """
    Performs a fused convolution + Mish activation.
    Weight is given in PyTorch's default layout [out_c, in_c, kH, kW];
    we reorder it to [out_c, kH, kW, in_c] for a simple flat pointer.
    """
    # Reorder weight to [out_c, k, k, in_c] (flattened later)
    w_reordered = conv_weight.permute(0, 2, 3, 1).contiguous()

    batch, _, h, w = x.shape
    k = conv_weight.shape[2]
    out_h = h - k + 1
    out_w = w - k + 1

    # Allocate output tensor
    out = torch.empty((batch, conv_weight.size(0), out_h, out_w),
                      dtype=x.dtype, device=x.device)

    # Launch the custom fused kernel (loop‑invariant hoisted)
    fused_ext.fused_conv(x, w_reordered, conv_bias, out,
                         subtract_value_1, subtract_value_2)

    return out
