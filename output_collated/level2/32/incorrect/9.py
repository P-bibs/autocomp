# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_012207/code_3.py
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

# -------------------------------------------------------------------------
# CUDA source – two kernels + host launcher
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// -------------------------------------------------------------------------
// Kernel 1: fused convolution + bias + scaling
// -------------------------------------------------------------------------
__global__ void conv_forward_kernel(
    const float* __restrict__ x,          // input  (B, C_in, H, W)
    const float* __restrict__ weight,     // weight (C_out, C_in, K, K)
    const float* __restrict__ bias,       // bias   (C_out)  or nullptr
    const int use_bias,                   // 1 if bias is present
    float* __restrict__ out,              // output (B, C_out, H_out, W_out)
    const int batch,
    const int in_c,
    const int out_c,
    const int height,
    const int width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const float scale_factor,
    const int out_h,
    const int out_w)
{
    // tile size = 16x16
    const int tile_h = 16;
    const int tile_w = 16;

    // total number of tiles per output plane
    const int tiles_y = (out_h + tile_h - 1) / tile_h;
    const int tiles_x = (out_w + tile_w - 1) / tile_w;
    const int total_tiles = tiles_y * tiles_x;

    // decode block index
    const int block_id = blockIdx.x;
    const int batch_id   = block_id / (out_c * total_tiles);
    const int rest       = block_id % (out_c * total_tiles);
    const int oc         = rest / total_tiles;               // output channel
    const int tile_id    = rest % total_tiles;
    const int tile_y     = tile_id / tiles_x;
    const int tile_x     = tile_id % tiles_x;

    // pixel coordinates inside the tile
    const int oh = tile_y * tile_h + threadIdx.y;
    const int ow = tile_x * tile_w + threadIdx.x;
    if (oh >= out_h || ow >= out_w) return;

    // ---- load weight for this output channel into shared memory ----
    extern __shared__ float weight_shared[];
    if (threadIdx.y == 0 && threadIdx.x == 0) {
        for (int ic = 0; ic < in_c; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int w_idx = ((oc * in_c + ic) * kernel_size + kh) * kernel_size + kw;
                    weight_shared[ic * kernel_size * kernel_size + kh * kernel_size + kw] = weight[w_idx];
                }
            }
        }
    }
    __syncthreads();

    // ---- convolution ----
    float sum = 0.0f;
    if (use_bias) sum = bias[oc];

    for (int ic = 0; ic < in_c; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            const int ih = oh * stride + kh * dilation - padding;
            if (ih < 0 || ih >= height) continue;
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int iw = ow * stride + kw * dilation - padding;
                if (iw < 0 || iw >= width) continue;
                const float w = weight_shared[ic * kernel_size * kernel_size + kh * kernel_size + kw];
                const float v = x[((batch_id * in_c + ic) * height + ih) * width + iw];
                sum += w * v;
            }
        }
    }

    sum *= scale_factor;
    out[((batch_id * out_c + oc) * out_h + oh) * out_w + ow] = sum;
}

// -------------------------------------------------------------------------
// Kernel 2: channel-wise minimum (min over output channels)
// -------------------------------------------------------------------------
__global__ void reduce_min_kernel(
    const float* __restrict__ conv_out,   // (B, C_out, H_out, W_out)
    float* __restrict__ min_out,          // (B, 1, H_out, W_out)
    const int batch,
    const int out_c,
    const int out_h,
    const int out_w)
{
    const int tile_h = 16;
    const int tile_w = 16;

    const int batch_id = blockIdx.z;
    const int tile_y   = blockIdx.y;
    const int tile_x   = blockIdx.x;

    const int oh = tile_y * tile_h + threadIdx.y;
    const int ow = tile_x * tile_w + threadIdx.x;
    if (oh >= out_h || ow >= out_w) return;

    float min_val = 1e38f;
    for (int oc = 0; oc < out_c; ++oc) {
        const float v = conv_out[((batch_id * out_c + oc) * out_h + oh) * out_w + ow];
        if (v < min_val) min_val = v;
    }
    min_out[(batch_id * out_h + oh) * out_w + ow] = min_val;
}

// -------------------------------------------------------------------------
// Host function – launches both kernels and returns the final tensor
// -------------------------------------------------------------------------
torch::Tensor fused_op(
    torch::Tensor x,          // (B, C_in, H, W)
    torch::Tensor weight,     // (C_out, C_in, K, K)
    torch::Tensor bias,       // (C_out)  – may be empty
    int stride,
    int padding,
    int dilation,
    int groups,
    double scale_factor)
{
    // tensors must already be on CUDA
    const int B      = x.size(0);
    const int C_in   = x.size(1);
    const int H      = x.size(2);
    const int W      = x.size(3);
    const int C_out  = weight.size(0);
    const int K      = weight.size(2);

    const int out_h = (H + 2*padding - dilation*(K-1) - 1) / stride + 1;
    const int out_w = (W + 2*padding - dilation*(K-1) - 1) / stride + 1;

    auto conv_out = torch::empty({B, C_out, out_h, out_w}, x.options());

    // ----- convolution kernel -----
    const int tile_h = 16, tile_w = 16;
    const int tiles_y = (out_h + tile_h - 1) / tile_h;
    const int tiles_x = (out_w + tile_w - 1) / tile_w;
    const int total_tiles = tiles_y * tiles_x;

    const int num_blocks_conv = B * C_out * total_tiles;
    dim3 block_conv(16, 16);               // 256 threads
    dim3 grid_conv(num_blocks_conv);

    const int shared_mem = C_in * K * K * sizeof(float);
    const int use_bias = (bias.numel() > 0) ? 1 : 0;

    conv_forward_kernel<<<grid_conv, block_conv, shared_mem>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        use_bias ? bias.data_ptr<float>() : nullptr,
        use_bias,
        conv_out.data_ptr<float>(),
        B, C_in, C_out, H, W, K,
        stride, padding, dilation, groups,
        static_cast<float>(scale_factor),
        out_h, out_w);

    // ----- reduction kernel -----
    auto min_out = torch::empty({B, 1, out_h, out_w}, x.options());

    dim3 block_red(16, 16);
    dim3 grid_red(tiles_x, tiles_y, B);   // 3-D grid: (tile_x, tile_y, batch)

    reduce_min_kernel<<<grid_red, block_red>>>(
        conv_out.data_ptr<float>(),
        min_out.data_ptr<float>(),
        B, C_out, out_h, out_w);

    return min_out;
}
"""

# -------------------------------------------------------------------------
# C++ interface – PyBind11 binding
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

torch::Tensor fused_op(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    double scale_factor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused convolution + min forward");
}
"""

# -------------------------------------------------------------------------
# Compile the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# -------------------------------------------------------------------------
# Re-implementation of functional_model using the custom CUDA extension
# -------------------------------------------------------------------------
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
    # Make sure all inputs reside on the GPU
    if not x.is_cuda:
        x = x.cuda()
    if not conv_weight.is_cuda:
        conv_weight = conv_weight.cuda()
    if conv_bias is not None and not conv_bias.is_cuda:
        conv_bias = conv_bias.cuda()

    # Call the fused CUDA kernel (conv + scale + min)
    out = fused_ext.fused_op(
        x,
        conv_weight,
        conv_bias,
        conv_stride,
        conv_padding,
        conv_dilation,
        conv_groups,
        scale_factor,
    )
    return out


# -------------------------------------------------------------------------
# Remaining code (unchanged, only for context)
# -------------------------------------------------------------------------
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
