# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_113642/code_2.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Optimized kernel using float4 vectorization
__global__ void fused_conv_transpose_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ conv_bias,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int IC, int OC, int IH, int IW, int OH, int OW, int KH, int KW) 
{
    // Simplified implicit GEMM approach for stride 1, padding 0
    int oc = blockIdx.z;
    int n = blockIdx.y;
    int oh = blockIdx.x;

    // Vectorized width processing (OW must be multiple of 4)
    for (int ow = threadIdx.x * 4; ow < OW; ow += blockDim.x * 4) {
        float vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        
        for (int ic = 0; ic < IC; ++ic) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    int ih = oh - kh;
                    int iw_base = ow - kw;
                    if (ih >= 0 && ih < IH) {
                        float w = weight[((oc * IC + ic) * KH + kh) * KW + kw];
                        #pragma unroll
                        for(int v=0; v<4; v++) {
                            int iw = iw_base + v;
                            if(iw >= 0 && iw < IW) {
                                vals[v] += input[((n * IC + ic) * IH + ih) * IW + iw] * w;
                            }
                        }
                    }
                }
            }
        }
        
        // Add convolution bias before applying tanh with secondary bias
        float4* out_ptr = (float4*)&output[(((n * OC + oc) * OH + oh) * OW + ow)];
        float4 res;
        res.x = tanhf((vals[0] + conv_bias[oc]) - bias[oc]);
        res.y = tanhf((vals[1] + conv_bias[oc]) - bias[oc]);
        res.z = tanhf((vals[2] + conv_bias[oc]) - bias[oc]);
        res.w = tanhf((vals[3] + conv_bias[oc]) - bias[oc]);
        *out_ptr = res;
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor conv_bias, torch::Tensor bias, torch::Tensor output) {
    int N = input.size(0), IC = input.size(1), IH = input.size(2), IW = input.size(3);
    int OC = weight.size(0), OH = output.size(2), OW = output.size(3);
    int KH = weight.size(2), KW = weight.size(3);
    
    dim3 grid(OH, N, OC);
    dim3 block(min(512, (OW + 3) / 4)); // Ensure we don't exceed max block size
    
    fused_conv_transpose_kernel<<<grid, block>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        conv_bias.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(), 
        N, IC, OC, IH, IW, OH, OW, KH, KW);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor conv_bias, torch::Tensor bias, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op_forward", &fused_op_forward, "Fused ConvTranspose + Bias + Tanh");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    bias,
):
    # Validate assumptions: stride=1, padding=0, dilation=1, groups=1
    assert conv_transpose_stride == (1, 1)
    assert conv_transpose_padding == (0, 0)
    assert conv_transpose_dilation == (1, 1)
    assert conv_transpose_groups == 1
    assert conv_transpose_output_padding == (0, 0)
    
    # Calculate output dimensions for stride 1, padding 0, dilation 1
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = conv_transpose_weight.shape
    OH = IH + KH - 1
    OW = IW + KW - 1
    
    # Ensure width is multiple of 4 for vectorization
    assert OW % 4 == 0
    
    # Allocate output tensor
    output = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)
    
    # Run custom fused kernel
    fused_ext.fused_op_forward(x, conv_transpose_weight, conv_transpose_bias, bias.view(-1), output)
    
    return output

# Setup for testing
batch_size = 32
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

