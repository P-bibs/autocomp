# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_082041/code_2.py
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

# Optimized CUDA kernel with shared memory tiling for convolution
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_WIDTH 16

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out, int H, int W, int K,
    int stride, int padding
) {
    extern __shared__ float shared_x[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int oh = blockIdx.y * TILE_WIDTH + ty;
    int ow = blockIdx.x * TILE_WIDTH + tx;
    int n = blockIdx.z;

    int OH = (H + 2 * padding - K) / stride + 1;
    int OW = (W + 2 * padding - K) / stride + 1;

    // Shared memory for input tile (including padding)
    // Size: (TILE_WIDTH + 2*(K-1)) x (TILE_WIDTH + 2*(K-1)) x C_in
    int tile_height = TILE_WIDTH + 2*(K-1);
    int tile_width = TILE_WIDTH + 2*(K-1);

    float min_val = 0.0f;
    bool valid_thread = (oh < OH && ow < OW);

    if (valid_thread) {
        // Load input tile to shared memory
        for (int ci = 0; ci < C_in; ++ci) {
            for (int i = ty; i < tile_height; i += TILE_WIDTH) {
                for (int j = tx; j < tile_width; j += TILE_WIDTH) {
                    int ih = oh * stride - padding + i;
                    int iw = ow * stride - padding + j;
                    
                    float val = 0.0f;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        val = x[((n * C_in + ci) * H + ih) * W + iw];
                    }
                    shared_x[(i * tile_width + j) * C_in + ci] = val;
                }
            }
            __syncthreads();
        }

        // Perform convolution and find minimum across channels
        bool first_channel = true;
        for (int co = 0; co < C_out; ++co) {
            float sum = bias[co];
            
            // Convolution using shared memory
            for (int ci = 0; ci < C_in; ++ci) {
                for (int kh = 0; kh < K; ++kh) {
                    for (int kw = 0; kw < K; ++kw) {
                        int sh_i = ty + kh;
                        int sh_j = tx + kw;
                        sum += shared_x[(sh_i * tile_width + sh_j) * C_in + ci] *
                               weight[(((co * C_in + ci) * K + kh) * K + kw)];
                    }
                }
            }
            
            // Update minimum value
            if (first_channel) {
                min_val = sum;
                first_channel = false;
            } else {
                min_val = fminf(min_val, sum);
            }
        }
        
        // Apply double tanh activation
        float res = tanhf(tanhf(min_val));
        output[((n * OH + oh) * OW + ow)] = res;
    }
}

void fused_conv_min_tanh_launcher(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding
) {
    int N = x.size(0);
    int C_in = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int C_out = weight.size(0);
    int K = weight.size(2);
    
    int OH = (H + 2 * padding - K) / stride + 1;
    int OW = (W + 2 * padding - K) / stride + 1;

    dim3 grid((OW + TILE_WIDTH - 1) / TILE_WIDTH,
              (OH + TILE_WIDTH - 1) / TILE_WIDTH,
              N);
    dim3 block(TILE_WIDTH, TILE_WIDTH);

    // Shared memory size: (TILE_WIDTH + 2*(K-1))^2 * C_in * sizeof(float)
    int tile_height = TILE_WIDTH + 2*(K-1);
    int tile_width = TILE_WIDTH + 2*(K-1);
    size_t shared_mem_size = tile_height * tile_width * C_in * sizeof(float);

    fused_conv_min_tanh_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out, H, W, K,
        stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_min_tanh_launcher(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_min_tanh", &fused_conv_min_tanh_launcher, "Fused Conv-Min-Tanh operation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_min_tanh_ext',
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
    # Validate inputs
    assert conv_dilation == 1, "Only dilation=1 supported"
    assert conv_groups == 1, "Only groups=1 supported"
    
    # Get output dimensions
    N, C_in, H, W = x.shape
    C_out, _, K, _ = conv_weight.shape
    OH = (H + 2 * conv_padding - K) // conv_stride + 1
    OW = (W + 2 * conv_padding - K) // conv_stride + 1
    
    # Create output tensor
    output = torch.empty((N, 1, OH, OW), device=x.device, dtype=x.dtype)
    
    # Launch optimized kernel
    fused_ext.fused_conv_min_tanh(x, conv_weight, conv_bias, output, conv_stride, conv_padding)
    
    return output
