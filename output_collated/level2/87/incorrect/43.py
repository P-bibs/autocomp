# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_145317/code_10.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_DIM 16

__device__ __forceinline__
float weight_at(const float* __restrict__ w, int oc, int ky, int kx, int ic, int K, int in_c) {
    return w[((ky * K + kx) * in_c) + ic];
}

__global__ void fused_conv_mish_shared_kernel(
    const float* __restrict__ input, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ output,
    int batch, int in_c, int in_h, int in_w, int out_c, int K,
    int out_h, int out_w, float sub1, float sub2) {
    
    const int tile_row = blockIdx.x;
    const int tile_col = blockIdx.y;
    const int oc_batch = blockIdx.z;
    
    const int oc = oc_batch % out_c;
    const int b  = oc_batch / out_c;
    
    const int th = threadIdx.y;
    const int tw = threadIdx.x;
    
    const int out_y = tile_row * TILE_DIM + th;
    const int out_x = tile_col * TILE_DIM + tw;
    
    if (out_y >= out_h || out_x >= out_w) return;
    
    extern __shared__ float shmem[];
    const int SMEM_PITCH = TILE_DIM + K - 1;
    
    const float* i_base = input + b * (in_c * in_h * in_w);
    const float* w_ptr  = weight + oc * (K * K * in_c);
    
    float acc = bias[oc];
    
    for (int ic = 0; ic < in_c; ++ic) {
        for (int dy = th; dy < SMEM_PITCH; dy += blockDim.y) {
            int in_y = tile_row * TILE_DIM + dy;
            for (int dx = tw; dx < SMEM_PITCH; dx += blockDim.x) {
                int in_x = tile_col * TILE_DIM + dx;
                float val = 0.f;
                if (in_y < in_h && in_x < in_w) {
                    val = i_base[ic * (in_h * in_w) + in_y * in_w + in_x];
                }
                shmem[dy * SMEM_PITCH + dx] = val;
            }
        }
        __syncthreads();

        for (int ky = 0; ky < K; ++ky) {
            const int sm_y = th + ky;
            for (int kx = 0; kx < K; ++kx) {
                const int sm_x = tw + kx;
                float iv = shmem[sm_y * SMEM_PITCH + sm_x];
                float wv = weight_at(w_ptr, 0, ky, kx, ic, K, in_c);
                acc += iv * wv;
            }
        }
        __syncthreads();
    }

    float val = acc - sub1 - sub2;
    float mish = val * tanhf(logf(1.0f + expf(val)));
    
    const int out_idx = ((b * out_c + oc) * out_h + out_y) * out_w + out_x;
    output[out_idx] = mish;
}

void fused_conv_mish_shared(
    int batch, int in_c, int in_h, int in_w,
    int out_c, int K,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    float sub1, float sub2) {
    
    int out_h = in_h - K + 1;
    int out_w = in_w - K + 1;

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(
        (out_h + TILE_DIM - 1) / TILE_DIM,
        (out_w + TILE_DIM - 1) / TILE_DIM,
        batch * out_c
    );

    size_t shmem_bytes = (TILE_DIM + K - 1) * (TILE_DIM + K - 1) * sizeof(float);

    fused_conv_mish_shared_kernel<<<grid, block, shmem_bytes>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, in_c, in_h, in_w,
        out_c, K,
        out_h, out_w,
        sub1, sub2
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_mish_shared(
    int batch, int in_c, int in_h, int in_w,
    int out_c, int K,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    float sub1, float sub2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &fused_conv_mish_shared, "Fused Convolution + Mish with Shared Memory");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride=1, conv_padding=0, 
                     conv_dilation=1, conv_groups=1, subtract_value_1, subtract_value_2):
    w_reordered = conv_weight.permute(0, 2, 3, 1).contiguous()
    
    batch, _, h, w = x.shape
    K = conv_weight.shape[2]
    out_h = h - K + 1
    out_w = w - K + 1
    out = torch.empty((batch, conv_weight.size(0), out_h, out_w), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_conv(
        batch, conv_weight.size(1), h, w,
        conv_weight.size(0), K,
        x, w_reordered, conv_bias, out,
        subtract_value_1, subtract_value_2
    )
    return out
