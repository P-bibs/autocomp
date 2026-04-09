# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_013320/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'scale_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'scale_factor']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, scales the output, and then applies a minimum operation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

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
    if 'scale_factor' in flat_state:
        state_kwargs['scale_factor'] = flat_state['scale_factor']
    else:
        state_kwargs['scale_factor'] = getattr(model, 'scale_factor')
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
# Helper functions required by the harness (unchanged)
# ----------------------------------------------------------------------
batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


# ----------------------------------------------------------------------
# Inline CUDA / C++ code
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

// ------------------------------------------------------------------
// Fused convolution + min-reduction + scale kernel (NCHW layout, supports groups)
// ------------------------------------------------------------------
__global__ void fused_conv_min_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int out_h,
    const int out_w,
    const float scale_factor)
{
    // One block per output spatial location (oh, ow) and batch
    int block_id = blockIdx.x;
    int b = block_id / (out_h * out_w);
    int rem = block_id % (out_h * out_w);
    int oh = rem / out_w;
    int ow = rem % out_w;

    int tid = threadIdx.x;
    int out_c_per_group = out_channels / groups;
    int in_c_per_group = in_channels / groups;

    // Shared memory for parallel reduction
    extern __shared__ float sdata[];
    
    float min_val = std::numeric_limits<float>::max();
    
    // Each thread handles multiple output channels if needed
    for (int oc = tid; oc < out_channels; oc += blockDim.x) {
        float sum = 0.0f;
        if (bias != nullptr) sum = bias[oc];

        int group_id = oc / out_c_per_group;
        int ic_start = group_id * in_c_per_group;

        for (int ic = 0; ic < in_c_per_group; ++ic) {
            int actual_ic = ic_start + ic;
            for (int kh = 0; kh < kernel_h; ++kh) {
                int iy = oh * stride + kh * dilation - padding;
                if (iy < 0 || iy >= height) continue;
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int ix = ow * stride + kw * dilation - padding;
                    if (ix < 0 || ix >= width) continue;
                    int in_idx = ((b * in_channels + actual_ic) * height + iy) * width + ix;
                    int wt_idx = ((oc * in_c_per_group + ic) * kernel_h + kh) * kernel_w + kw;
                    sum += input[in_idx] * weight[wt_idx];
                }
            }
        }
        if (sum < min_val) min_val = sum;
    }

    sdata[tid] = min_val;
    __syncthreads();

    // Block-wide parallel reduction to find minimum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] < sdata[tid]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the result scaled by scale_factor
    if (tid == 0) {
        int out_idx = ((b * 1 + 0) * out_h + oh) * out_w + ow;
        output[out_idx] = sdata[0] * scale_factor;
    }
}

// ------------------------------------------------------------------
// Host wrapper (called from Python)
// ------------------------------------------------------------------
void fused_conv_min_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding,
    int dilation,
    int groups,
    int out_h,
    int out_w,
    float scale_factor)
{
    const int threads = 256;  // Heuristic value, can be tuned
    const int blocks = batch * out_h * out_w;
    // Shared memory size: one float per thread
    fused_conv_min_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch, in_channels, out_channels, height, width,
        kernel_h, kernel_w,
        stride, padding, dilation, groups,
        out_h, out_w,
        scale_factor);
    cudaDeviceSynchronize();
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_min_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding,
    int dilation,
    int groups,
    int out_h,
    int out_w,
    float scale_factor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_min_forward", &fused_conv_min_forward, "Fused convolution, min-reduction, and scale forward");
}
"""

# Compile the fused extension
fused_ext = load_inline(
    name="fused_op",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
)


# ----------------------------------------------------------------------
# Optimised functional_model (replaces the original)
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
    scale_factor,
):
    # ------------------------------------------------------------------
    # Move tensors to the GPU and ensure contiguity
    # ------------------------------------------------------------------
    if not x.is_cuda:
        x = x.cuda()
    conv_weight = conv_weight.to(x.device)
    if conv_bias is not None:
        conv_bias = conv_bias.to(x.device)
    else:
        # Create an empty tensor – the kernel will treat it as "no bias"
        conv_bias = torch.empty(0, dtype=x.dtype, device=x.device)

    x = x.contiguous()
    conv_weight = conv_weight.contiguous()
    conv_bias = conv_bias.contiguous()

    # ------------------------------------------------------------------
    # Compute output spatial dimensions
    # ------------------------------------------------------------------
    batch, in_c, H, W = x.shape
    out_c = conv_weight.shape[0]
    kernel_h = conv_weight.shape[2]
    kernel_w = conv_weight.shape[3]

    out_h = (H + 2 * conv_padding - conv_dilation * (kernel_h - 1) - 1) // conv_stride + 1
    out_w = (W + 2 * conv_padding - conv_dilation * (kernel_w - 1) - 1) // conv_stride + 1

    # ------------------------------------------------------------------
    # Single fused kernel: convolution + min-reduction + scaling
    # ------------------------------------------------------------------
    out = torch.empty((batch, 1, out_h, out_w), dtype=x.dtype, device=x.device)
    fused_ext.fused_conv_min_forward(
        x,
        conv_weight,
        conv_bias,
        out,
        batch,
        in_c,
        out_c,
        H,
        W,
        kernel_h,
        kernel_w,
        conv_stride,
        conv_padding,
        conv_dilation,
        conv_groups,
        out_h,
        out_w,
        float(scale_factor),
    )
    return out
