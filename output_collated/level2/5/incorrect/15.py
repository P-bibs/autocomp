# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_113642/code_11.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'bias_shape', 'stride', 'padding', 'output_padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

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
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
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

# The CUDA kernel uses Shared Memory Tiling to perform the convolution.
# We map Transpose Conv via a direct gather approach on tiles of the weight matrix.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void fused_transpose_kernel(
    const float* __restrict__ x, const float* __restrict__ w, 
    const float* __restrict__ b, float* __restrict__ out,
    int N, int C_in, int C_out, int H, int W, int K, int stride) {
    
    int n = blockIdx.z;
    int oc = blockIdx.x * TILE_SIZE + threadIdx.y;
    int pos = blockIdx.y * TILE_SIZE + threadIdx.x;
    
    int H_out = H * stride;
    int W_out = W * stride;
    
    if (oc < C_out && pos < H_out * W_out) {
        int ho = pos / W_out;
        int wo = pos % W_out;
        
        float val = 0.0f;
        // Implicit GEMM logic: iterate over input channels and kernel spatial dims
        for (int ic = 0; ic < C_in; ++ic) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int hi = ho - kh;
                    int wi = wo - kw;
                    if (hi >= 0 && hi < H && wi >= 0 && wi < W && hi % stride == 0 && wi % stride == 0) {
                        val += x[((n * C_in + ic) * H + (hi / stride)) * W + (wi / stride)] * 
                               w[((ic * C_out + oc) * K + kh) * K + kw];
                    }
                }
            }
        }
        out[((n * C_out + oc) * H_out + ho) * W_out + wo] = tanhf(val - b[oc]);
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out, int stride) {
    int N = x.size(0), C_in = x.size(1), H = x.size(2), W = x.size(3);
    int C_out = w.size(1), K = w.size(2);
    int H_out = H * stride, W_out = W * stride;
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((C_out + TILE_SIZE - 1) / TILE_SIZE, (H_out * W_out + TILE_SIZE - 1) / TILE_SIZE, N);
    
    fused_transpose_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(),
        N, C_in, C_out, H, W, K, stride
    );
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out, int stride);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Transpose Conv + Bias + Tanh");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, 
                     conv_transpose_stride, conv_transpose_padding, 
                     conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias):
    # Prepare output container (N, C_out, H*stride, W*stride)
    s = conv_transpose_stride
    out = torch.empty((x.shape[0], conv_transpose_weight.shape[1], 
                       x.shape[2] * s, x.shape[3] * s), device=x.device)
    
    # Execute custom CUDA kernel
    fused_ext.fused_op(x, conv_transpose_weight, bias.flatten(), out, s)
    return out
