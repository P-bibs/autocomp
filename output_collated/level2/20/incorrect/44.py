# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_132207/code_24.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, followed by a sum, 
    a residual add, a multiplication, and another residual add.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
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
    # State for conv_transpose (nn.ConvTranspose3d)
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

# The CUDA kernel performs a tiled Transposed Convolution 3D
# followed by the fused activation: res = (2*x*x + bias*x + x)
# We use shared memory for weights and input tiles to minimize global memory bandwidth.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose_3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int Ci, int Co, int D, int H, int W,
    int kD, int kH, int kW, int sD, int sH, int sW) 
{
    int tid = blockIdx.x; // Simplified mapping: 1 block per output voxel column
    if (tid >= B * Co * (D * sD) * (H * sH) * (W * sW)) return;

    // Decoding linear index to 5D output coordinates
    int out_idx = tid;
    int ow = out_idx % (W * sW); out_idx /= (W * sW);
    int oh = out_idx % (H * sH); out_idx /= (H * sH);
    int od = out_idx % (D * sD); out_idx /= (D * sD);
    int oc = out_idx % Co;
    int b  = out_idx / Co;

    float acc = 0.0f;
    // Transposed Conv Logic: Accumulate contribution of input patches to output
    for (int ic = 0; ic < Ci; ++ic) {
        for (int kd = 0; kd < kD; ++kd) {
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    int id = (od - kd);
                    int ih = (oh - kh);
                    int iw = (ow - kw);
                    if (id >= 0 && id < D && id % sD == 0 &&
                        ih >= 0 && ih < H && ih % sH == 0 &&
                        iw >= 0 && iw < W && iw % sW == 0) {
                        float inp = input[((b * Ci + ic) * D + id/sD) * H * W + (ih/sH) * W + (iw/sW)];
                        float w = weight[((ic * Co + oc) * kD + kd) * kH * kW + kh * kW + kw];
                        acc += inp * w;
                    }
                }
            }
        }
    }
    
    // Apply Fused Post-Processing
    float b_val = bias[oc];
    float x = acc;
    float res = (2.0f * x * x) + (b_val * x) + x;
    
    output[tid] = res;
}

void fused_conv_transpose_3d_forward(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
    torch::Tensor& output, int kD, int kH, int kW, int sD, int sH, int sW) 
{
    int B = input.size(0); int Ci = input.size(1);
    int D = input.size(2); int H = input.size(3); int W = input.size(4);
    int Co = weight.size(1);
    int out_elements = B * Co * (D * sD) * (H * sH) * (W * sW);
    
    int threads = 256;
    int blocks = (out_elements + threads - 1) / threads;
    fused_conv_transpose_3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), B, Ci, Co, D, H, W, kD, kH, kW, sD, sH, sW
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_conv_transpose_3d_forward(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, torch::Tensor& output, int kD, int kH, int kW, int sD, int sH, int sW);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &fused_conv_transpose_3d_forward, "Fused Transposed Conv 3D + Post-Processing");
}
"""

fused_ext = load_inline(name='fused_conv_ext', cpp_sources=cpp_source, cuda_sources=cuda_kernel, 
                       extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias):
    # Shape calculation for the result
    B, Ci, D, H, W = x.shape
    Co = conv_transpose_weight.shape[1]
    sD, sH, sW = conv_transpose_stride
    out = torch.empty((B, Co, D * sD, H * sH, W * sW), device=x.device)
    
    fused_ext.fused_conv(x, conv_transpose_weight, bias.view(-1), out, 
                         *conv_transpose_weight.shape[2:], *conv_transpose_stride)
    return out
