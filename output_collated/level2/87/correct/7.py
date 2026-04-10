# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_141921/code_12.py
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

# ------------------------------------------------------------
#  CUDA kernel – tiled version
# ------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_H 8
#define TILE_W 8

extern "C"
__global__ void fused_conv_mish_tiled_kernel(
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

    const int bz = blockIdx.z;
    const int b  = bz / out_c;
    const int oc = bz % out_c;

    const int out_x0 = blockIdx.x * TILE_W;
    const int out_y0 = blockIdx.y * TILE_H;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int in_tile_h = TILE_H + k - 1;
    const int in_tile_w = TILE_W + k - 1;
    const int in_tile_elems = in_tile_h * in_tile_w;

    extern __shared__ float shmem[];

    // Load input tiles for all channels into shared memory
    for (int ic = 0; ic < in_c; ++ic) {
        float* sh_input = shmem + ic * in_tile_elems;
        for (int y = ty; y < in_tile_h; y += blockDim.y) {
            const int in_y = out_y0 + y;
            for (int x = tx; x < in_tile_w; x += blockDim.x) {
                const int in_x = out_x0 + x;
                if (in_y < in_h && in_x < in_w) {
                    sh_input[y * in_tile_w + x] = input[((b * in_c + ic) * in_h + in_y) * in_w + in_x];
                } else {
                    sh_input[y * in_tile_w + x] = 0.0f;
                }
            }
        }
    }
    __syncthreads();

    const int out_x = out_x0 + tx;
    const int out_y = out_y0 + ty;

    if (out_x < out_w && out_y < out_h) {
        float acc = bias[oc];

        for (int ic = 0; ic < in_c; ++ic) {
            const float* sh_input = shmem + ic * in_tile_elems;
            const float* w_ptr = weight + (oc * in_c + ic) * k * k;
            for (int i = 0; i < k; ++i) {
                const int sh_row = (ty + i) * in_tile_w;
                for (int j = 0; j < k; ++j) {
                    acc += sh_input[sh_row + (tx + j)] * w_ptr[i * k + j];
                }
            }
        }

        float val = acc - sub1 - sub2;
        float mish = val * tanhf(logf(1.0f + expf(val)));
        output[((b * out_c + oc) * out_h + out_y) * out_w + out_x] = mish;
    }
}

void fused_conv_mish_tiled(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output, float sub1, float sub2)
{
    const int batch = input.size(0);
    const int in_c = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_c = weight.size(0);
    const int k = weight.size(2);
    const int out_h = in_h - k + 1;
    const int out_w = in_w - k + 1;

    dim3 block(TILE_W, TILE_H);
    dim3 grid((out_w + TILE_W - 1) / TILE_W, (out_h + TILE_H - 1) / TILE_H, batch * out_c);

    size_t shmem_bytes = size_t(in_c) * size_t((TILE_H + k - 1) * (TILE_W + k - 1)) * sizeof(float);

    fused_conv_mish_tiled_kernel<<<grid, block, shmem_bytes>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch, in_c, in_h, in_w, out_c, k, sub1, sub2);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_conv_mish_tiled(torch::Tensor i, torch::Tensor w, torch::Tensor b, torch::Tensor o, float s1, float s2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_mish_tiled", &fused_conv_mish_tiled, "Fused Conv/Sub/Mish");
}
"""

fused_ext = load_inline(
    name='fused_ext_tiled', 
    cpp_sources=cpp_source, 
    cuda_sources=cuda_source, 
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride=1, conv_padding=0, 
                     conv_dilation=1, conv_groups=1, subtract_value_1, subtract_value_2):
    batch, _, h, w = x.shape
    k = conv_weight.shape[2]
    out = torch.empty((batch, conv_weight.size(0), h - k + 1, w - k + 1), device=x.device, dtype=x.dtype)
    fused_ext.fused_conv_mish_tiled(x, conv_weight, conv_bias, out, float(subtract_value_1), float(subtract_value_2))
    return out
