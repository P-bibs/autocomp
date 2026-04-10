# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_154319/code_10.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'add_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'add_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

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
    if 'add_value' in flat_state:
        state_kwargs['add_value'] = flat_state['add_value']
    else:
        state_kwargs['add_value'] = getattr(model, 'add_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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

# Optimized CUDA Kernel: 
# Uses a tiled approach to parallelize over N, OC, OH, OW. 
# Implements the transconv logic implicitly to avoid memory-heavy Im2Col.
# Fuses the activation chain (add, min, gelu, mul) into the kernel.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_transpose_kernel(
    const float* __restrict__ input, const float* __restrict__ weight, 
    float* __restrict__ output, float add_val, float mul_val,
    int N, int IC, int IH, int IW, int OC, int K, int OH, int OW, int stride, int padding) {
    
    int n = blockIdx.z / OC;
    int oc = blockIdx.z % OC;
    int oh = blockIdx.y;
    int ow = blockIdx.x;
    
    if (n < N && oc < OC && oh < OH && ow < OW) {
        float sum = 0.0f;
        
        // Implicit GEMM: Each output pixel is a dot product of weights and a patch of the input.
        // For Transposed Conv, the input index is derived from the output index.
        for (int ic = 0; ic < IC; ++ic) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int ih_full = oh + padding - kh;
                    int iw_full = ow + padding - kw;
                    
                    if (ih_full % stride == 0 && iw_full % stride == 0) {
                        int ih = ih_full / stride;
                        int iw = iw_full / stride;
                        
                        if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                            float in_val = input[((n * IC + ic) * IH + ih) * IW + iw];
                            float w_val = weight[((ic * OC + oc) * K + kh) * K + kw];
                            sum += in_val * w_val;
                        }
                    }
                }
            }
        }
        
        float val = fminf(sum + add_val, 0.0f);
        output[((n * OC + oc) * OH + oh) * OW + ow] = fast_gelu(val) * mul_val;
    }
}

void fused_conv_transpose_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor output, 
                                  float add_val, float mul_val, int stride, int padding) {
    int N = input.size(0); int IC = input.size(1);
    int IH = input.size(2); int IW = input.size(3);
    int OC = weight.size(1); int K = weight.size(2);
    int OH = output.size(2); int OW = output.size(3);
    
    dim3 threads(1, 1, 1);
    dim3 blocks(OW, OH, N * OC);
    fused_conv_transpose_kernel<<<blocks, threads>>>(input.data_ptr<float>(), weight.data_ptr<float>(), 
                                                     output.data_ptr<float>(), add_val, mul_val, 
                                                     N, IC, IH, IW, OC, K, OH, OW, stride, padding);
}
"""

cpp_source = r"""
void fused_conv_transpose_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor output, float add_val, float mul_val, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose", &fused_conv_transpose_forward, "Fused ConvTranspose + Ops");
}
"""

fused_ext = load_inline(
    name='fused_conv_transpose', 
    cpp_sources=cpp_source, 
    cuda_sources=cuda_kernel, 
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, add_value, multiply_value):
    # Calculate output dimensions
    stride = conv_transpose_stride[0]
    padding = conv_transpose_padding[0]
    H_out = (x.size(2) - 1) * stride - 2 * padding + (conv_transpose_weight.size(2) - 1) + conv_transpose_output_padding[0] + 1
    W_out = (x.size(3) - 1) * stride - 2 * padding + (conv_transpose_weight.size(3) - 1) + conv_transpose_output_padding[1] + 1
    
    out = torch.empty((x.size(0), conv_transpose_weight.size(1), H_out, W_out), device='cuda')
    
    # Run custom fused kernel
    fused_ext.fused_conv_transpose(x, conv_transpose_weight, out, float(add_value), float(multiply_value), stride, padding)
    
    # Adding bias if present
    if conv_transpose_bias is not None:
        out += conv_transpose_bias.view(1, -1, 1, 1)
        
    return out
