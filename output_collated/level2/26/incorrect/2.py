# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_040131/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x', 'add_input']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
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

# Optimization: Fusing ConvTranspose3D, Addition, and Hardswish into a single kernel
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>

#define THREADS_PER_BLOCK 256

__device__ __forceinline__ float hardswish_impl(float x) {
    return x * fmaxf(0.0f, fminf(1.0f, (x + 3.0f) / 6.0f));
}

__global__ void fused_conv_tr_add_hswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    int N, int IC, int OC,
    int ID, int IH, int IW,
    int OD, int OH, int OW,
    int KD, int KH, int KW,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * OC * OD * OH * OW;
    
    if (idx >= total_elements) return;
    
    // Decode output position
    int ow = idx % OW;
    int oh = (idx / OW) % OH;
    int od = (idx / (OW * OH)) % OD;
    int oc = (idx / (OW * OH * OD)) % OC;
    int n = idx / (OW * OH * OD * OC);
    
    // Conv transpose calculation
    float conv_val = 0.0f;
    
    // Iterate through kernel
    for (int kd = 0; kd < KD; ++kd) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                // Calculate corresponding input position
                int id = od * stride_d - padding_d + kd * dilation_d;
                int ih = oh * stride_h - padding_h + kh * dilation_h;
                int iw = ow * stride_w - padding_w + kw * dilation_w;
                
                // Check bounds
                if (id >= 0 && id < ID && ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                    for (int ic = 0; ic < IC; ++ic) {
                        if (oc * IC + ic < OC * IC) { // group check
                            // Weight index: [oc][ic][kd][kh][kw]
                            int w_idx = oc * (IC * KD * KH * KW) + 
                                       ic * (KD * KH * KW) + 
                                       kd * (KH * KW) + 
                                       kh * KW + kw;
                            
                            // Input index: [n][ic][id][ih][iw]
                            int i_idx = n * (IC * ID * IH * IW) + 
                                       ic * (ID * IH * IW) + 
                                       id * (IH * IW) + 
                                       ih * IW + iw;
                            
                            conv_val += input[i_idx] * weight[w_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add bias
    conv_val += bias[oc];
    
    // Add input
    int add_idx = n * (OC * OD * OH * OW) + 
                  oc * (OD * OH * OW) + 
                  od * (OH * OW) + 
                  oh * OW + ow;
                  
    float result = conv_val + add_input[add_idx];
    
    // Apply hardswish
    output[add_idx] = result * hardswish_impl(result);
}

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor add_input,
    torch::Tensor output,
    int N, int IC, int OC,
    int ID, int IH, int IW,
    int OD, int OH, int OW,
    int KD, int KH, int KW,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups
) {
    int total_elements = N * OC * OD * OH * OW;
    int blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    fused_conv_tr_add_hswish_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        add_input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w,
        groups
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor add_input,
    torch::Tensor output,
    int N, int IC, int OC,
    int ID, int IH, int IW,
    int OD, int OH, int OW,
    int KD, int KH, int KW,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused ConvTranspose3D + Add + Hardswish");
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

def functional_model(
    x,
    add_input,
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
    # Extract dimensions
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = conv_transpose_weight.shape
    
    # Calculate output dimensions
    stride_d, stride_h, stride_w = conv_transpose_stride
    padding_d, padding_h, padding_w = conv_transpose_padding
    dilation_d, dilation_h, dilation_w = conv_transpose_dilation
    
    OD = (ID - 1) * stride_d - 2 * padding_d + dilation_d * (KD - 1) + 1 + conv_transpose_output_padding[0]
    OH = (IH - 1) * stride_h - 2 * padding_h + dilation_h * (KH - 1) + 1 + conv_transpose_output_padding[1]
    OW = (IW - 1) * stride_w - 2 * padding_w + dilation_w * (KW - 1) + 1 + conv_transpose_output_padding[2]
    
    # Create output tensor
    output = torch.empty((N, OC, OD, OH, OW), dtype=x.dtype, device=x.device)
    
    # Call fused kernel
    fused_ext.fused_op(
        x, conv_transpose_weight, conv_transpose_bias, add_input, output,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w,
        conv_transpose_groups
    )
    
    return output

batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W), torch.rand(batch_size, out_channels, D*stride, H*stride, W*stride)]
