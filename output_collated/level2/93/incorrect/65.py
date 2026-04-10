# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_160727/code_6.py
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

# -------------------------------------------------------------------------
# CUDA source: custom transposed convolution fused with add/min/gelu/mul
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Fast GELU approximation used by many high‑performance kernels
__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Kernel: one block per output channel.
//   * weight tile (in_channels * Kh * Kw) is loaded into shared memory once.
//   * Grid‑stride loop iterates over all (batch, height, width) positions.
//   * Each thread computes the convolution for its position, then applies the
//     fused add → min(0) → GELU → mul chain.
__global__ void fused_conv_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int padding,
    const int dilation,
    const int has_bias,
    const float add_val,
    const float mul_val) {

    // One block = one output channel
    const int oc = blockIdx.x;

    // Shared memory for the weight tile of this output channel
    extern __shared__ float weight_cache[];
    const int weight_size = in_channels * kernel_h * kernel_w;
    const int oc_offset = oc * weight_size;

    // Load weight tile (coalesced across threads)
    for (int idx = threadIdx.x; idx < weight_size; idx += blockDim.x) {
        weight_cache[idx] = weight[oc_offset + idx];
    }
    __syncthreads();

    const int total_positions = batch * out_height * out_width;
    const int stride_loop = blockDim.x * gridDim.x;   // 256 * out_channels

    // Grid‑stride loop over all (batch, oh, ow) positions for this channel
    for (int pos = threadIdx.x; pos < total_positions; pos += stride_loop) {
        // Decode linear position into (batch, oh, ow)
        int n = pos / (out_height * out_width);
        int rem = pos % (out_height * out_width);
        int oh = rem / out_width;
        int ow = rem % out_width;

        // Start accumulation with optional bias
        float sum = has_bias ? bias[oc] : 0.0f;

        // ---- Transposed convolution (gather from input) ----
        // Unroll the inner loops for the known small kernel size (4x4)
        #pragma unroll
        for (int ic = 0; ic < in_channels; ++ic) {
            const int w_base = ic * kernel_h * kernel_w;
            #pragma unroll
            for (int kh = 0; kh < kernel_h; ++kh) {
                int i_h = oh + padding - kh * dilation;
                if (i_h % stride != 0) continue;
                i_h /= stride;
                if (i_h < 0 || i_h >= in_height) continue;
                #pragma unroll
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int i_w = ow + padding - kw * dilation;
                    if (i_w % stride != 0) continue;
                    i_w /= stride;
                    if (i_w < 0 || i_w >= in_width) continue;

                    float w = weight_cache[w_base + kh * kernel_w + kw];
                    int in_idx = ((n * in_channels + ic) * in_height + i_h) * in_width + i_w;
                    float inp = __ldg(&input[in_idx]);
                    sum += inp * w;
                }
            }
        }

        // ---- Fused add → min(0) → GELU → mul ----
        float val = sum + add_val;
        val = fminf(val, 0.0f);          // ReLU‑like clamp
        val = fast_gelu(val);
        val = val * mul_val;

        // Write final output
        int out_idx = ((n * out_channels + oc) * out_height + oh) * out_width + ow;
        output[out_idx] = val;
    }
}

// Host wrapper that launches the kernel with the correct shared‑memory size
void fused_conv_transpose_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding,
    int dilation,
    int has_bias,
    float add_val,
    float mul_val) {

    const int weight_size = in_channels * kernel_h * kernel_w;
    const int shared_mem = weight_size * sizeof(float);
    const int threads = 256;
    const int blocks = out_channels;               // one block per output channel

    fused_conv_transpose_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, in_channels, out_channels,
        in_height, in_width,
        out_height, out_width,
        kernel_h, kernel_w,
        stride, padding, dilation,
        has_bias,
        add_val, mul_val);
}
"""

# -------------------------------------------------------------------------
# C++ binding (pybind11) for the CUDA kernel
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding,
    int dilation,
    int has_bias,
    float add_val,
    float mul_val);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose", &fused_conv_transpose_forward,
          "Fused transposed convolution + add → min → gelu → mul");
}
"""

# -------------------------------------------------------------------------
# Compile the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_conv_transpose',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional model that uses the custom fused kernel
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
    add_value,
    multiply_value,
):
    # -----------------------------------------------------------------
    # Compute output spatial size (standard transposed‑conv formula)
    # -----------------------------------------------------------------
    kernel_h = conv_transpose_weight.shape[2]
    kernel_w = conv_transpose_weight.shape[3]

    out_h = (x.size(2) - 1) * conv_transpose_stride \
            - 2 * conv_transpose_padding \
            + conv_transpose_dilation * (kernel_h - 1) \
            + conv_transpose_output_padding + 1

    out_w = (x.size(3) - 1) * conv_transpose_stride \
            - 2 * conv_transpose_padding \
            + conv_transpose_dilation * (kernel_w - 1) \
            + conv_transpose_output_padding + 1

    # Allocate output tensor
    output = torch.empty(
        (x.size(0), conv_transpose_weight.shape[0], out_h, out_w),
        device='cuda', dtype=torch.float32
    )

    # Prepare bias (empty tensor if none – kernel checks has_bias flag)
    if conv_transpose_bias is None:
        bias = torch.empty(0, device='cuda', dtype=torch.float32)
        has_bias = 0
    else:
        bias = conv_transpose_bias
        has_bias = 1

    # -----------------------------------------------------------------
    # Launch the fused transposed‑conv + element‑wise kernel
    # -----------------------------------------------------------------
    fused_ext.fused_conv_transpose(
        x,                     # input  (N, C_in, H, W)
        conv_transpose_weight, # weight (C_out, C_in, Kh, Kw)
        bias,                  # bias   (C_out) or empty
        output,                # output (N, C_out, H_out, W_out)
        x.size(0),             # batch
        x.size(1),             # in_channels
        conv_transpose_weight.shape[0],   # out_channels
        x.size(2),             # in_height
        x.size(3),             # in_width
        out_h,                 # out_height
        out_w,                 # out_width
        kernel_h,
        kernel_w,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_dilation,
        has_bias,
        float(add_value),
        float(multiply_value)
    )
    return output

# -------------------------------------------------------------------------
# Helper functions used by the benchmark harness
# -------------------------------------------------------------------------
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
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
