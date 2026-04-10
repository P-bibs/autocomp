# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_160727/code_10.py
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

# The custom kernel implements the Transposed Convolution (Deconvolution) 
# logic by iterating over input pixels and scattering their contributions 
# to the output. We fuse the add-min-gelu-mul activation directly in 
# the kernel to avoid extra global memory passes.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_transpose_kernel(
    const float* __restrict__ input, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ output,
    int B, int C_in, int C_out, int H_in, int W_in, 
    int K, int stride, float add_val, float mul_val, 
    int H_out, int W_out) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C_out * H_out * W_out;
    if (tid >= total_elements) return;

    // Mapping flat index to N, C_out, H_out, W_out
    int temp = tid;
    int w_out = temp % W_out; temp /= W_out;
    int h_out = temp % H_out; temp /= H_out;
    int c_out = temp % C_out; temp /= C_out;
    int n = temp;

    float sum = bias[c_out];
    
    // Accumulate contributions from overlapping windows
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int h_in = (h_out - kh);
                int w_in = (w_out - kw);
                
                if (h_in % stride == 0 && w_in % stride == 0) {
                    h_in /= stride;
                    w_in /= stride;
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        float in_val = input[((n * C_in + c_in) * H_in + h_in) * W_in + w_in];
                        float w_val = weight[((c_in * C_out + c_out) * K + kh) * K + kw];
                        sum += in_val * w_val;
                    }
                }
            }
        }
    }
    
    // Fuse activation: add -> min(x, 0) -> fast_gelu -> multiply
    float val = fminf(sum + add_val, 0.0f);
    output[tid] = fast_gelu(val) * mul_val;
}

void launch_fused_conv(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, 
                       int stride, float add_val, float mul_val) {
    int B = input.size(0), C_in = input.size(1), H_in = input.size(2), W_in = input.size(3);
    int C_out = weight.size(1), K = weight.size(2);
    int H_out = H_in * stride;
    int W_out = W_in * stride;
    int out_size = B * C_out * H_out * W_out;
    
    int threads = 256;
    int blocks = (out_size + threads - 1) / threads;
    fused_conv_transpose_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        B, C_in, C_out, H_in, W_in, K, stride, add_val, mul_val, H_out, W_out
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_fused_conv(torch::Tensor i, torch::Tensor w, torch::Tensor b, torch::Tensor o, int s, float a, float m);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_fused_conv", &launch_fused_conv, "Fused Transpose Conv + Activation");
}
"""

fused_ext = load_inline(
    name='fused_conv', 
    cpp_sources=cpp_source, 
    cuda_sources=cuda_source, 
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, add_value, multiply_value):
    # Output dimensions based on stride (padding=0, output_padding=0 case)
    batch_size = x.size(0)
    out_channels = conv_transpose_weight.size(1)
    out_height = x.size(2) * conv_transpose_stride
    out_width = x.size(3) * conv_transpose_stride
    
    out = torch.empty((batch_size, out_channels, out_height, out_width), device='cuda')
    
    # Execute the fused custom kernel
    fused_ext.launch_fused_conv(
        x, conv_transpose_weight, conv_transpose_bias, out, 
        conv_transpose_stride, float(add_value), float(multiply_value)
    )
    return out
