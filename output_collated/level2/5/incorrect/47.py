# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_121229/code_7.py
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

# Optimization: Merge low-level operations (conv_transpose + sub + tanh) 
# using a fused CUDA kernel to minimize memory latency.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ sub_bias,
    float* __restrict__ output,
    int B, int C, int H, int W, 
    int OC, int K_H, int K_W,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int dilation_h, int dilation_w,
    int groups) {
    
    int out_W = (W - 1) * stride_w - 2 * padding_w + dilation_w * (K_W - 1) + 1 + output_padding_w;
    int out_H = (H - 1) * stride_h - 2 * padding_h + dilation_h * (K_H - 1) + 1 + output_padding_h;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_idx >= B * OC * out_H * out_W) return;

    int tmp = out_idx;
    int w_out = tmp % out_W; tmp /= out_W;
    int h_out = tmp % out_H; tmp /= out_H;
    int oc = tmp % OC;      tmp /= OC;
    int b = tmp;

    float acc = 0.0f;
    
    // Assuming groups = 1 for simplicity as in the original setup
    for (int ic = 0; ic < C; ++ic) {
        for (int kh = 0; kh < K_H; ++kh) {
            for (int kw = 0; kw < K_W; ++kw) {
                int h_in = (h_out + padding_h - kh * dilation_h);
                int w_in = (w_out + padding_w - kw * dilation_w);
                
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W && 
                    h_in % stride_h == 0 && w_in % stride_w == 0) {
                    
                    int h_idx = h_in / stride_h;
                    int w_idx = w_in / stride_w;
                    
                    acc += input[((b * C + ic) * H + h_idx) * W + w_idx] * 
                           weight[((oc * C + ic) * K_H + kh) * K_W + kw];
                }
            }
        }
    }
    
    // Add conv bias
    acc += conv_bias[oc];
    
    // Subtract sub bias and apply tanh
    float sub_bias_val = sub_bias[oc];
    output[out_idx] = tanhf(acc - sub_bias_val);
}

void fused_op_forward(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor conv_bias, 
    torch::Tensor sub_bias, 
    torch::Tensor output,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int dilation_h, int dilation_w,
    int groups) {
    
    int B = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int OC = weight.size(0);
    int K_H = weight.size(2);
    int K_W = weight.size(3);

    int out_W = (W - 1) * stride_w - 2 * padding_w + dilation_w * (K_W - 1) + 1 + output_padding_w;
    int out_H = (H - 1) * stride_h - 2 * padding_h + dilation_h * (K_H - 1) + 1 + output_padding_h;
    int total_elements = B * OC * out_H * out_W;
    
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_conv_transpose_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        conv_bias.data_ptr<float>(),
        sub_bias.data_ptr<float>(),
        output.data_ptr<float>(), 
        B, C, H, W, OC, K_H, K_W,
        stride_h, stride_w,
        padding_h, padding_w,
        output_padding_h, output_padding_w,
        dilation_h, dilation_w,
        groups);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor conv_bias, 
    torch::Tensor sub_bias, 
    torch::Tensor output,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int dilation_h, int dilation_w,
    int groups);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused ConvTranspose2d + Sub + Tanh");
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
    # Handle different input types for stride, padding, etc.
    if isinstance(conv_transpose_stride, int):
        stride_h = stride_w = conv_transpose_stride
    else:
        stride_h, stride_w = conv_transpose_stride
    
    if isinstance(conv_transpose_padding, int):
        padding_h = padding_w = conv_transpose_padding
    else:
        padding_h, padding_w = conv_transpose_padding
        
    if isinstance(conv_transpose_output_padding, int):
        output_padding_h = output_padding_w = conv_transpose_output_padding
    else:
        output_padding_h, output_padding_w = conv_transpose_output_padding
        
    if isinstance(conv_transpose_dilation, int):
        dilation_h = dilation_w = conv_transpose_dilation
    else:
        dilation_h, dilation_w = conv_transpose_dilation

    B, C, H, W = x.shape
    OC = conv_transpose_weight.size(0)
    K_H, K_W = conv_transpose_weight.size(2), conv_transpose_weight.size(3)

    # Calculate output dimensions
    out_H = (H - 1) * stride_h - 2 * padding_h + dilation_h * (K_H - 1) + 1 + output_padding_h
    out_W = (W - 1) * stride_w - 2 * padding_w + dilation_w * (K_W - 1) + 1 + output_padding_w
    
    output = torch.empty((B, OC, out_H, out_W), device=x.device, dtype=x.dtype)
    
    # The bias provided in the original code is (OC, 1, 1), reshape for kernel simplicity
    fused_ext.fused_op(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias.view(-1), 
        bias.view(-1),
        output,
        stride_h, stride_w,
        padding_h, padding_w,
        output_padding_h, output_padding_w,
        dilation_h, dilation_w,
        conv_transpose_groups
    )
    return output

batch_size = 32
in_channels  = 64  
out_channels = 64  
height = width = 256 
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
