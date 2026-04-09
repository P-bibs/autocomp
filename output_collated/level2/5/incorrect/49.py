# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_121229/code_10.py
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

# CUDA Kernel: Fuses conv transpose, bias subtraction, and Tanh
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_conv_transpose_bias_tanh_kernel(
    const float* input,       // [N, Ci, Hi, Wi]
    const float* weight,      // [Ci, Co/g, K, K] (transposed format)
    const float* bias,        // [Co]
    float* output,            // [N, Co, Ho, Wo]
    int N, int Ci, int Hi, int Wi,
    int Co, int K, int stride, int padding, int output_padding, int groups, int dilation,
    int Ho, int Wo
) {
    // Each thread computes one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = N * Co * Ho * Wo;
    
    if (idx >= total_outputs) return;
    
    // Calculate output indices
    int n = idx / (Co * Ho * Wo);
    int co = (idx / (Ho * Wo)) % Co;
    int ho = (idx / Wo) % Ho;
    int wo = idx % Wo;
    
    // Map to group
    int g = co / (Co / groups);
    int co_in_group = co % (Co / groups);
    
    float sum = 0.0f;
    
    // Iterate through kernel elements
    for (int ci_in_group = 0; ci_in_group < (Ci / groups); ++ci_in_group) {
        int ci = g * (Ci / groups) + ci_in_group;
        
        // Iterate through kernel dimensions
        for (int ky = 0; ky < K; ++ky) {
            for (int kx = 0; kx < K; ++kx) {
                // Calculate input position that contributes to this output
                int hi = ho * stride - padding + ky * dilation;
                int wi = wo * stride - padding + kx * dilation;
                
                // Check bounds
                if (hi >= 0 && hi < Hi && wi >= 0 && wi < Wi) {
                    // Weight index: [Ci, Co/g, K, K]
                    int weight_idx = ci * (Co/groups) * K * K + 
                                    co_in_group * K * K + 
                                    ky * K + kx;
                    
                    // Input index: [N, Ci, Hi, Wi]
                    int input_idx = n * Ci * Hi * Wi + 
                                   ci * Hi * Wi + 
                                   hi * Wi + wi;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Apply bias and tanh
    float result = tanhf(sum - bias[co]);
    
    // Write output
    output[idx] = result;
}

void fused_conv_transpose_bias_tanh_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
) {
    int N = input.size(0);
    int Ci = input.size(1);
    int Hi = input.size(2);
    int Wi = input.size(3);
    
    int Co = weight.size(1) * groups;  // Adjusted for transposed conv
    int K = weight.size(2);
    
    // Calculate output dimensions
    int Ho = (Hi - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1;
    int Wo = (Wi - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1;
    
    int total_elements = N * Co * Ho * Wo;
    
    // Launch configuration
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_conv_transpose_bias_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, Ci, Hi, Wi,
        Co, K, stride, padding, output_padding, groups, dilation,
        Ho, Wo
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_bias_tanh_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_bias_tanh_forward", &fused_conv_transpose_bias_tanh_forward, "Fused Conv Transpose + Bias + Tanh forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose_ext',
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
    # Calculate output dimensions
    N, Ci, Hi, Wi = x.shape
    Co = conv_transpose_weight.size(0)
    K = conv_transpose_weight.size(2)
    
    stride = conv_transpose_stride[0] if isinstance(conv_transpose_stride, (list, tuple)) else conv_transpose_stride
    padding = conv_transpose_padding[0] if isinstance(conv_transpose_padding, (list, tuple)) else conv_transpose_padding
    output_padding = conv_transpose_output_padding[0] if isinstance(conv_transpose_output_padding, (list, tuple)) else conv_transpose_output_padding
    dilation = conv_transpose_dilation[0] if isinstance(conv_transpose_dilation, (list, tuple)) else conv_transpose_dilation
    
    Ho = (Hi - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
    Wo = (Wi - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
    
    # Create output tensor
    output = torch.empty(N, Co, Ho, Wo, dtype=x.dtype, device=x.device)
    
    # Run fused kernel
    fused_ext.fused_conv_transpose_bias_tanh_forward(
        x, conv_transpose_weight, conv_transpose_bias, output,
        stride, padding, output_padding, conv_transpose_groups, dilation
    )
    
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

