# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_145317/code_27.py
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

# -----------------------------------------------------------------------
# CUDA kernel that caches the per‑output‑channel weight in shared memory
# -----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#ifndef TILE_X
#define TILE_X 16
#endif
#ifndef TILE_Y
#define TILE_Y 16
#endif

// Mish activation: x * tanh(softplus(x))
__device__ __forceinline__ float mish(float x) {
    return x * tanhf(log1pf(expf(x)));
}

extern "C"
__global__ void fused_conv_mish_shared_kernel(
        const float* __restrict__ input,
        const float* __restrict__ weight,
        const float* __restrict__ bias,
        float*       __restrict__ output,
        const int B,
        const int C_in,
        const int H,
        const int W,
        const int C_out,
        const int k,
        const float sub1,
        const float sub2,
        const int H_out,
        const int W_out) {

    extern __shared__ float sh_weight[];

    const int bz = blockIdx.z;
    const int b  = bz / C_out;
    const int oc = bz % C_out;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int out_x = blockIdx.x * TILE_X + tx;
    const int out_y = blockIdx.y * TILE_Y + ty;

    // Load weights into shared memory once per block
    const int weight_elems = k * k * C_in;
    const int tid = ty * TILE_X + tx;
    for (int i = tid; i < weight_elems; i += (TILE_X * TILE_Y)) {
        sh_weight[i] = weight[oc * weight_elems + i];
    }
    __syncthreads();

    if (out_x >= W_out || out_y >= H_out) return;

    float acc = bias[oc];
    const int in_base = ((b * C_in) * H + out_y) * W + out_x;

    #pragma unroll
    for (int i = 0; i < k; ++i) {
        #pragma unroll
        for (int j = 0; j < k; ++j) {
            const int in_offset = i * W + j;
            const int w_base = (i * k + j) * C_in;
            #pragma unroll
            for (int ic = 0; ic < C_in; ++ic) {
                acc += input[in_base + ic * H * W + in_offset] * sh_weight[w_base + ic];
            }
        }
    }

    output[((b * C_out + oc) * H_out + out_y) * W_out + out_x] = mish(acc - sub1 - sub2);
}

void fused_conv_mish_shared(
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor bias,
        torch::Tensor output,
        float sub1,
        float sub2) {

    const int B = input.size(0);
    const int C_in = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int C_out = weight.size(0);
    const int k = weight.size(2);
    const int H_out = H - k + 1;
    const int W_out = W - k + 1;

    const dim3 blockDim(16, 16);
    const dim3 gridDim((W_out + 15) / 16, (H_out + 15) / 16, B * C_out);
    const size_t shmem_bytes = k * k * C_in * sizeof(float);

    fused_conv_mish_shared_kernel<<<gridDim, blockDim, shmem_bytes>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), B, C_in, H, W, C_out, k, sub1, sub2, H_out, W_out
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_conv_mish_shared(torch::Tensor i, torch::Tensor w, torch::Tensor b, torch::Tensor o, float s1, float s2);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &fused_conv_mish_shared, "Fused Conv + Mish");
}
"""

fused_ext = load_inline(
    name='fused_shared_ext', cpp_sources=cpp_source, cuda_sources=cuda_kernel, 
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride=1, conv_padding=0, 
                     conv_dilation=1, conv_groups=1, subtract_value_1, subtract_value_2):
    # Weight layout expectation: [out_c, k, k, in_c]
    w_reordered = conv_weight.permute(0, 2, 3, 1).contiguous()
    
    batch, _, h, w = x.shape
    k = conv_weight.shape[2]
    out = torch.empty((batch, conv_weight.size(0), h - k + 1, w - k + 1), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_conv(x, w_reordered, conv_bias, out, float(subtract_value_1), float(subtract_value_2))
    return out
