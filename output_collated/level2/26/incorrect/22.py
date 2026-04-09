# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_042434/code_3.py
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

# The fused CUDA kernel performs: ConvTranspose3D -> Add -> Hardswish
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__device__ __forceinline__ float hardswish(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

__global__ void fused_conv_transpose3d_add_hardswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    int N, int IC, int OC,
    int ID, int IH, int IW,
    int OD, int OH, int OW,
    int KD, int KH, int KW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    // Calculate global thread index
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * OC * OD * OH * OW;
    
    if (out_idx >= total_elements) return;
    
    // Decompose output index
    int ow = out_idx % OW;
    int oh = (out_idx / OW) % OH;
    int od = (out_idx / (OW * OH)) % OD;
    int oc = (out_idx / (OW * OH * OD)) % OC;
    int n  = out_idx / (OW * OH * OD * OC);
    
    float sum = 0.0f;
    
    // Iterate through input channels and kernel
    for (int ic = 0; ic < IC; ++ic) {
        // Compute the corresponding input region
        // For convolution transpose, we need to find which input pixels contribute to this output pixel
        int kd_start = (od + pad_d - (KD - 1) * dilation_d) % stride_d;
        int kh_start = (oh + pad_h - (KH - 1) * dilation_h) % stride_h;
        int kw_start = (ow + pad_w - (KW - 1) * dilation_w) % stride_w;
        
        for (int kd = kd_start; kd < KD; kd += stride_d) {
            for (int kh = kh_start; kh < KH; kh += stride_h) {
                for (int kw = kw_start; kw < KW; kw += stride_w) {
                    // Calculate input coordinates
                    int id = (od + pad_d - kd * dilation_d) / stride_d;
                    int ih = (oh + pad_h - kh * dilation_h) / stride_h;
                    int iw = (ow + pad_w - kw * dilation_w) / stride_w;
                    
                    // Check bounds
                    if (id >= 0 && id < ID && ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                        // Calculate indices
                        int input_idx = ((((n * IC + ic) * ID + id) * IH + ih) * IW + iw);
                        int weight_idx = (((((ic * OC + oc) * KD + kd) * KH + kh) * KW + kw));
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add bias
    sum += conv_bias[oc];
    
    // Add residual
    sum += add_input[out_idx];
    
    // Apply hardswish
    output[out_idx] = hardswish(sum);
}

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor add_input,
    torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    const int N = input.size(0);
    const int IC = input.size(1);
    const int ID = input.size(2);
    const int IH = input.size(3);
    const int IW = input.size(4);
    
    const int OC = weight.size(1);
    const int KD = weight.size(2);
    const int KH = weight.size(3);
    const int KW = weight.size(4);
    
    const int OD = add_input.size(2);
    const int OH = add_input.size(3);
    const int OW = add_input.size(4);
    
    const int total_elements = N * OC * OD * OH * OW;
    
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    fused_conv_transpose3d_add_hardswish_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        add_input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor add_input,
    torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused ConvTranspose3D, Add, Hardswish");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
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
    # Validate groups (only supporting groups=1 for this implementation)
    if conv_transpose_groups != 1:
        raise ValueError("Only groups=1 is supported in this implementation")
    
    # Handle stride, padding, dilation
    if isinstance(conv_transpose_stride, int):
        stride_d = stride_h = stride_w = conv_transpose_stride
    else:
        stride_d, stride_h, stride_w = conv_transpose_stride
    
    if isinstance(conv_transpose_padding, int):
        pad_d = pad_h = pad_w = conv_transpose_padding
    else:
        pad_d, pad_h, pad_w = conv_transpose_padding
        
    if isinstance(conv_transpose_dilation, int):
        dilation_d = dilation_h = dilation_w = conv_transpose_dilation
    else:
        dilation_d, dilation_h, dilation_w = conv_transpose_dilation
    
    # Create output tensor
    output = torch.empty_like(add_input)
    
    # Call fused kernel
    fused_ext.fused_op(
        x, conv_transpose_weight, conv_transpose_bias, add_input, output,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w
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
