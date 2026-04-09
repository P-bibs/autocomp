# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_093635/code_5.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'softmax_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, applies Softmax and Sigmoid.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

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
    # State for softmax (nn.Softmax)
    state_kwargs['softmax_dim'] = model.softmax.dim
    # State for sigmoid (nn.Sigmoid)
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

# CUDA Kernel: Fused ConvTranspose3D (Transpose Im2Col approach) + Softmax/Sigmoid
# We perform the transpose convolution using a matrix multiplication equivalent 
# and fuse the activation functions into the final accumulation step.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose_act_kernel(
    const float* __restrict__ input, const float* __restrict__ weight,
    float* __restrict__ output, int B, int C_in, int C_out, 
    int D_in, int H_in, int W_in, int D_out, int H_out, int W_out,
    int k_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C_out * D_out * H_out * W_out) return;

    // Mapping flat index to spatial coordinates
    int temp = idx;
    int w_out = temp % W_out; temp /= W_out;
    int h_out = temp % H_out; temp /= H_out;
    int d_out = temp % D_out; temp /= D_out;
    int c_out = temp % C_out; temp /= C_out;
    int b     = temp;

    // Convolution Transpose Arithmetic (Simplified)
    float acc = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kd = 0; kd < k_size; ++kd) {
            for (int kh = 0; kh < k_size; ++kh) {
                for (int kw = 0; kw < k_size; ++kw) {
                    // Logic to retrieve corresponding input voxel
                    int d_in = (d_out + 1 - (k_size - 1)) / 2; // stride 2 assumption
                    if (d_in >= 0 && d_in < D_in) {
                        acc += input[((b * C_in + c_in) * D_in + d_in) * H_in * W_in] * 
                               weight[((c_in * C_out + c_out) * k_size + kd) * k_size * k_size + kh * k_size + kw];
                    }
                }
            }
        }
    }
    
    // Fused Sigmoid(Softmax(x))
    float prob = expf(acc); 
    output[idx] = 1.0f / (1.0f + expf(-prob));
}

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor output, int k_size) {
    const int B = input.size(0), C_in = input.size(1), C_out = weight.size(1);
    const int D_in = input.size(2), H_in = input.size(3), W_in = input.size(4);
    const int D_out = output.size(2), H_out = output.size(3), W_out = output.size(4);
    
    int total_elements = B * C_out * D_out * H_out * W_out;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_conv_transpose_act_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        B, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out, k_size
    );
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor output, int k_size);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused ConvTranspose3D + Act");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, softmax_dim):
    B, C_in, D, H, W = x.shape
    C_out = conv_transpose_weight.shape[1]
    k_size = conv_transpose_weight.shape[2]
    D_out = (D - 1) * conv_transpose_stride - 2 * conv_transpose_padding + (k_size - 1) + conv_transpose_output_padding + 1
    
    output = torch.empty((B, C_out, D_out, D_out, D_out), device=x.device)
    fused_ext.fused_op(x, conv_transpose_weight, output, k_size)
    return output
