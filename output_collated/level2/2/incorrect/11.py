# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_162535/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape', 'scaling_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias', 'scaling_factor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

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
    if 'scaling_factor' in flat_state:
        state_kwargs['scaling_factor'] = flat_state['scaling_factor']
    else:
        state_kwargs['scaling_factor'] = getattr(model, 'scaling_factor')
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

# --- CUDA Kernel ---
# We implement a fused ConvTranspose2d + Bias + Clamp/Scale/Clamp sequence.
# To keep this focused, we use an implicit-gemm approach in the kernel.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose_kernel(
    const float* __restrict__ x, const float* __restrict__ weight, 
    const float* __restrict__ conv_bias, const float* __restrict__ bias,
    float* __restrict__ out,
    int B, int Ci, int Co, int Hi, int Wi, int Ho, int Wo, 
    int K, int stride, int padding, int output_padding, float scale) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B * Co * Ho * Wo) return;

    int tmp = tid;
    int w = tmp % Wo; tmp /= Wo;
    int h = tmp % Ho; tmp /= Ho;
    int co = tmp % Co; tmp /= Co;
    int b = tmp;

    float acc = conv_bias[co] + bias[co]; // Fuse both biases
    // ConvTranspose2d logic
    for (int ci = 0; ci < Ci; ++ci) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                // For transpose conv: we map output position back to input 
                // with stride and kernel size
                // out[h][w] += sum over kernel positions of weight[kh][kw] * input[ih][iw]
                // where (h,w) = (ih*stride + kh - padding), solving for ih,iw:
                int ih = h + padding - kh;
                int iw = w + padding - kw;
                if (ih % stride == 0 && iw % stride == 0) {
                    ih /= stride;
                    iw /= stride;
                    if (ih >= 0 && ih < Hi && iw >= 0 && iw < Wi) {
                        acc += x[((b * Ci + ci) * Hi + ih) * Wi + iw] * weight[((co * Ci + ci) * K + kh) * K + kw];
                    }
                }
            }
        }
    }

    // Fused element-wise operations
    float val = fminf(fmaxf(acc, 0.0f), 1.0f); // Clamp [0,1]
    val *= scale;
    val = fminf(fmaxf(val, 0.0f), 1.0f);       // Clamp [0,1] again
    out[tid] = val / scale;
}

void fused_op_forward(
    int blocks, int threads,
    torch::Tensor x, torch::Tensor weight, torch::Tensor conv_bias, torch::Tensor bias,
    torch::Tensor out, int K, int stride, int padding, int output_padding, float scale) {
    
    int B = x.size(0), Ci = x.size(1), Hi = x.size(2), Wi = x.size(3);
    int Co = weight.size(0);
    int Ho = out.size(2);
    int Wo = out.size(3);
    
    fused_conv_transpose_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), conv_bias.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), B, Ci, Co, Hi, Wi, Ho, Wo, K, stride, padding, output_padding, scale);
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the function in the .cu file
void fused_op_forward(
    int blocks, int threads,
    torch::Tensor x, torch::Tensor weight, torch::Tensor conv_bias, torch::Tensor bias,
    torch::Tensor out, int K, int stride, int padding, int output_padding, float scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused ConvTranspose2d + Bias + Clamp/Scale/Clamp");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)


def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias, scaling_factor):
    B, Ci, Hi, Wi = x.shape
    Co, _, K, _ = conv_transpose_weight.shape
    
    # Output dimensions from original ConvTranspose2d formula
    Ho = (Hi - 1) * conv_transpose_stride + (K - 1) * conv_transpose_dilation + 1 - 2*conv_transpose_padding + conv_transpose_output_padding
    Wo = (Wi - 1) * conv_transpose_stride + (K - 1) * conv_transpose_dilation + 1 - 2*conv_transpose_padding + conv_transpose_output_padding

    out = torch.empty((B, Co, Ho, Wo), device=x.device, dtype=x.dtype)
    
    threads = 256
    blocks = (B * Co * Ho * Wo + threads - 1) // threads

    fused_ext.fused_op(
        blocks, threads,
        x, conv_transpose_weight, conv_transpose_bias, bias,
        out, K, conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding, scaling_factor
    )
    
    return out


# Constants for test input (as provided)
batch_size = 128
in_channels  = 64  
out_channels = 64  
height = width = 128 
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
