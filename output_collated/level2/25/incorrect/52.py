# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_090933/code_8.py
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

# --- C++ Interface / Bindings ---
cpp_source = r"""
#include <torch/extension.h>

void launch_fused_conv_min_tanh(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_fused_conv_min_tanh, "Fused Conv + Min + Tanh Kernel");
}
"""

# --- CUDA Kernel Implementation ---
# We use shared memory to hold the weights and perform computation per (n, oh, ow).
# The 2080Ti has 48KB of shared memory per SM. 
# 64 channels * 16 channels * 3 * 3 floats = 9216 floats (~36KB), fits perfectly.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

#define TILE_H 8
#define TILE_W 8

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out,
    int H, int W,
    int K,
    int stride,
    int padding,
    int OH, int OW
) {
    int oh = blockIdx.y * TILE_H + threadIdx.y;
    int ow = blockIdx.x * TILE_W + threadIdx.x;
    int n = blockIdx.z;

    if (oh >= OH || ow >= OW) return;

    // Shared memory for weights: [C_out][C_in][K][K]
    // Allocated dynamically via kernel launch configuration
    extern __shared__ float s_weight[];

    int weight_size = C_out * C_in * K * K;
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < weight_size; i += blockDim.x * blockDim.y) {
        s_weight[i] = weight[i];
    }
    __syncthreads();

    float min_val = 1e38f; // Representing infinity

    // Fused Conv + Min
    for (int co = 0; co < C_out; ++co) {
        float sum = bias[co];
        int weight_offset = co * C_in * K * K;
        
        for (int ci = 0; ci < C_in; ++ci) {
            for (int kh = 0; kh < K; ++kh) {
                int ih = oh * stride + kh - padding;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < K; ++kw) {
                    int iw = ow * stride + kw - padding;
                    if (iw >= 0 && iw < W) {
                        float val_x = x[((n * C_in + ci) * H + ih) * W + iw];
                        float val_w = s_weight[weight_offset + (ci * K * K) + (kh * K + kw)];
                        sum += val_x * val_w;
                    }
                }
            }
        }
        if (sum < min_val) min_val = sum;
    }

    // Double Tanh
    float t1 = tanhf(min_val);
    output[((n * OH) + oh) * OW + ow] = tanhf(t1);
}

void launch_fused_conv_min_tanh(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding
) {
    int N = input.size(0);
    int C_in = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int C_out = weight.size(0);
    int K = weight.size(2);

    int OH = (H + 2 * padding - K) / stride + 1;
    int OW = (W + 2 * padding - K) / stride + 1;

    dim3 block(TILE_W, TILE_H);
    dim3 grid((OW + TILE_W - 1) / TILE_W, (OH + TILE_H - 1) / TILE_H, N);
    
    size_t shared_mem_size = C_out * C_in * K * K * sizeof(float);

    fused_conv_min_tanh_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, C_in, C_out, H, W, K, stride, padding, OH, OW
    );
}
"""

# Compile Extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

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
    # Only supports dilation=1, groups=1 as per specialized optimization
    OH = (x.shape[2] + 2 * conv_padding - conv_weight.shape[2]) // conv_stride + 1
    OW = (x.shape[3] + 2 * conv_padding - conv_weight.shape[3]) // conv_stride + 1
    
    output = torch.empty((x.shape[0], 1, OH, OW), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, conv_weight, conv_bias, output, conv_stride, conv_padding)
    return output
