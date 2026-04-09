# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_121229/code_24.py
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

# Optimized CUDA kernel using float4 vectorization
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void fused_bias_tanh_vectorized_kernel(float* __restrict__ data, const float* __restrict__ bias, 
                                                  int N, int C, int H, int W) {
    int total_elements = N * C * H * W;
    int hw = H * W;
    
    // Process 4 elements at a time
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x * 4) {
        if (i + 3 < total_elements) {
            float4* data_ptr = reinterpret_cast<float4*>(&data[i]);
            float4 val = *data_ptr;
            
            // Unrolling processing for each of the 4 elements
            // Note: Since bias varies per channel, we calculate indices for each element
            float out[4];
            float inputs[4] = {val.x, val.y, val.z, val.w};
            
            #pragma unroll
            for(int j = 0; j < 4; ++j) {
                int curr_idx = i + j;
                int c = (curr_idx / hw) % C;
                out[j] = tanhf(inputs[j] - bias[c]);
            }
            
            *data_ptr = make_float4(out[0], out[1], out[2], out[3]);
        } else {
            // Handle cleanup for non-multiple-of-4 sizes
            for (int j = i; j < total_elements; ++j) {
                int c = (j / hw) % C;
                data[j] = tanhf(data[j] - bias[c]);
            }
        }
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor bias) {
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    
    int total_elements = N * C * H * W;
    int threads = 256;
    // Process 4 elements per thread
    int blocks = (total_elements / 4 + threads - 1) / threads;
    blocks = std::min(blocks, 65535);

    fused_bias_tanh_vectorized_kernel<<<blocks, threads>>>(x.data_ptr<float>(), bias.data_ptr<float>(), N, C, H, W);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor x, torch::Tensor bias);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Bias Subtraction and Tanh Vectorized");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
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
    import torch.nn.functional as F
    
    # Requirement 6: Replace standard convolution if necessary.
    # Instruction 1 says "Keep F.conv_transpose2d() as-is - It's a highly optimized operation".
    # We prioritize the instruction to keep conv_transpose2d given it's a PyTorch primitive.
    x = F.conv_transpose2d(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        stride=conv_transpose_stride, 
        padding=conv_transpose_padding, 
        output_padding=conv_transpose_output_padding, 
        groups=conv_transpose_groups, 
        dilation=conv_transpose_dilation
    )
    
    # Ensure optimal memory alignment for float4 loads
    x = x.contiguous()
    bias_flat = bias.view(-1).contiguous()
    
    fused_ext.fused_op(x, bias_flat)
    
    return x

# Placeholder parameters
batch_size = 32
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
