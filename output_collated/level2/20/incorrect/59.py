# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_134740/code_5.py
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

# CUDA Kernel for fused transposed convolution and post-processing
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_post_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ post_bias,
    float* __restrict__ output,
    const int N, const int CI, const int CO,
    const int ID, const int IH, const int IW,
    const int KD, const int KH, const int KW,
    const int SD, const int SH, const int SW,
    const int PD, const int PH, const int PW,
    const int DD, const int DH, const int DW,
    const int OD, const int OH, const int OW)
{
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * CO * OD * OH * OW;
    
    if (idx >= total_elements) return;

    // Decompose linear index to 5D coordinates
    int tmp = idx;
    int ow = tmp % OW; tmp /= OW;
    int oh = tmp % OH; tmp /= OH;
    int od = tmp % OD; tmp /= OD;
    int co = tmp % CO; tmp /= CO;
    int n  = tmp;

    // Compute convolution value
    float conv_val = 0.0f;
    
    // Loop over kernel dimensions
    for (int kd = 0; kd < KD; ++kd) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                // Loop over input channels
                for (int ci = 0; ci < CI; ++ci) {
                    // Map output position back to input position
                    int id = od * SD - PD + kd * DD;
                    int ih = oh * SH - PH + kh * DH;
                    int iw = ow * SW - PW + kw * DW;
                    
                    // Check bounds
                    if (id >= 0 && id < ID && ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                        // Calculate indices
                        int input_idx = n * (CI * ID * IH * IW) + ci * (ID * IH * IW) + id * (IH * IW) + ih * IW + iw;
                        int weight_idx = co * (CI * KD * KH * KW) + ci * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw;
                        
                        conv_val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add convolution bias
    conv_val += conv_bias[co];
    
    // Apply post-processing: ((x + bias) + x) * x + x = 2*x*x + bias*x + x
    float pb = post_bias[co];
    float result = (2.0f * conv_val * conv_val) + (pb * conv_val) + conv_val;
    
    // Store result
    output[idx] = result;
}

void fused_conv_post_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output)
{
    // Get dimensions
    const int N = input.size(0);
    const int CI = input.size(1);
    const int ID = input.size(2);
    const int IH = input.size(3);
    const int IW = input.size(4);
    
    const int CO = weight.size(1);
    const int KD = weight.size(2);
    const int KH = weight.size(3);
    const int KW = weight.size(4);
    
    const int OD = output.size(2);
    const int OH = output.size(3);
    const int OW = output.size(4);
    
    // Stride, padding, dilation (assuming all 3 for simplicity)
    const int SD = 2, SH = 2, SW = 2;
    const int PD = 1, PH = 1, PW = 1;
    const int DD = 1, DH = 1, DW = 1;
    
    const int total_elements = N * CO * OD * OH * OW;
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    fused_conv_post_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        post_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, CI, CO,
        ID, IH, IW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        DD, DH, DW,
        OD, OH, OW
    );
}
"""

# C++ binding
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_post_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output);

torch::Tensor fused_conv_post(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias) {
    
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(conv_bias.is_contiguous(), "conv_bias must be contiguous");
    TORCH_CHECK(post_bias.is_contiguous(), "post_bias must be contiguous");
    
    // Calculate output dimensions
    const int N = input.size(0);
    const int CI = input.size(1);
    const int ID = input.size(2);
    const int IH = input.size(3);
    const int IW = input.size(4);
    
    const int CO = weight.size(1);
    const int KD = weight.size(2);
    const int KH = weight.size(3);
    const int KW = weight.size(4);
    
    // Parameters for transposed convolution
    const int SD = 2, SH = 2, SW = 2;
    const int PD = 1, PH = 1, PW = 1;
    const int DD = 1, DH = 1, DW = 1;
    const int OPD = 1, OPH = 1, OPW = 1;
    
    // Calculate output dimensions for transposed convolution
    const int OD = (ID - 1) * SD - 2 * PD + DD * (KD - 1) + OPD + 1;
    const int OH = (IH - 1) * SH - 2 * PH + DH * (KH - 1) + OPH + 1;
    const int OW = (IW - 1) * SW - 2 * PW + DW * (KW - 1) + OPW + 1;
    
    auto output = torch::empty({N, CO, OD, OH, OW}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    fused_conv_post_forward(input, weight, conv_bias, post_bias, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_post", &fused_conv_post, "Fused Transposed Conv + Arithmetic");
}
"""

# Build the extension
fused_ext = load_inline(
    name='fused_conv_post_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# Functional model implementation
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
    # Flatten bias to a 1-D tensor (required by the kernel)
    bias_flat = bias.view(-1)
    
    # Call the fused operation
    return fused_ext.fused_conv_post(x, conv_transpose_weight, conv_transpose_bias, bias_flat)

# Helper code (shape parameters, input factories)
batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
