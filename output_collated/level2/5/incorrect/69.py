# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_123254/code_5.py
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

#define TILE_SIZE 16
#define THREADS_PER_BLOCK 256

__global__ void conv_transpose_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    float* __restrict__ output,
    const float* __restrict__ eltwise_bias,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int stride, int padding
) {
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + TILE_SIZE * TILE_SIZE;
    
    int hw_out = blockIdx.x * blockDim.x + threadIdx.x;
    if (hw_out >= H_out * W_out) return;
    
    int h_out = hw_out / W_out;
    int w_out = hw_out % W_out;
    
    for (int n = 0; n < N; n++) {
        for (int c_out = 0; c_out < C_out; c_out++) {
            float acc = 0.0f;
            
            for (int c_in = 0; c_in < C_in; c_in++) {
                for (int kh = 0; kh < K; kh++) {
                    for (int kw = 0; kw < K; kw++) {
                        int h_in = h_out + padding - kh;
                        int w_in = w_out + padding - kw;
                        
                        if (h_in % stride == 0 && w_in % stride == 0) {
                            h_in /= stride;
                            w_in /= stride;
                            
                            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                                float input_val = input[((n * C_in + c_in) * H_in + h_in) * W_in + w_in];
                                float weight_val = weight[(((c_out * C_in) + c_in) * K + kh) * K + kw];
                                acc += input_val * weight_val;
                            }
                        }
                    }
                }
            }
            
            float conv_result = acc + conv_bias[c_out];
            float final_result = tanhf(conv_result - eltwise_bias[c_out]);
            output[((n * C_out + c_out) * H_out + h_out) * W_out + w_out] = final_result;
        }
    }
}

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor output,
    torch::Tensor eltwise_bias,
    int stride,
    int padding,
    int K
) {
    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int C_out = conv_bias.size(0);
    int H_out = output.size(2);
    int W_out = output.size(3);
    
    int threads = THREADS_PER_BLOCK;
    int blocks = (H_out * W_out + threads - 1) / threads;
    
    size_t shared_mem_size = (TILE_SIZE * TILE_SIZE * 2) * sizeof(float);
    
    conv_transpose_fused_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        eltwise_bias.data_ptr<float>(),
        N, C_in, H_in, W_in, C_out, H_out, W_out, K, stride, padding
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor output,
    torch::Tensor eltwise_bias,
    int stride,
    int padding,
    int K
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op_forward", &fused_op_forward, "Fused ConvTranspose2d + Bias + Tanh");
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
    # Validate that groups is 1 and dilation is (1, 1) as our kernel assumes these
    if conv_transpose_groups != 1:
        raise ValueError("Only conv_transpose_groups=1 is supported")
    if conv_transpose_dilation != (1, 1):
        raise ValueError("Only conv_transpose_dilation=(1, 1) is supported")
        
    # Shape inference for output
    N, C_in, H_in, W_in = x.shape
    C_out = conv_transpose_bias.shape[0]
    kernel_size = conv_transpose_weight.size(2)
    
    H_out = (H_in - 1) * conv_transpose_stride + conv_transpose_output_padding + kernel_size - 2 * conv_transpose_padding
    W_out = (W_in - 1) * conv_transpose_stride + conv_transpose_output_padding + kernel_size - 2 * conv_transpose_padding
    
    output = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Call the fused kernel
    fused_ext.fused_op_forward(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        output, 
        bias.view(-1), 
        conv_transpose_stride, 
        conv_transpose_padding,
        kernel_size
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
