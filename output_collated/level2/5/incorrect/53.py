# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_121229/code_15.py
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

# ----------------------------------------------------------------------
# Inline CUDA code – implements the fused transposed-conv + bias + tanh
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ----------------------------------------------------------------------
// Transposed convolution kernel
// ----------------------------------------------------------------------
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    float* __restrict__ output,
    const int N, const int C_in, const int H_in, const int W_in,
    const int C_out, const int K_h, const int K_w,
    const int H_out, const int W_out,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int out_pad_h, const int out_pad_w,
    const int dilation_h, const int dilation_w,
    const int groups) {
    
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out_elements = N * C_out * H_out * W_out;
    
    if (out_idx >= total_out_elements) return;
    
    int n = out_idx / (C_out * H_out * W_out);
    int remaining = out_idx % (C_out * H_out * W_out);
    int c_out = remaining / (H_out * W_out);
    remaining = remaining % (H_out * W_out);
    int h_out = remaining / W_out;
    int w_out = remaining % W_out;
    
    // Determine which group this output channel belongs to
    int group_id = c_out * groups / C_out;
    int c_out_in_group = c_out % (C_out / groups);
    
    float sum = 0.0f;
    
    // Iterate through kernel positions
    for (int kh = 0; kh < K_h; ++kh) {
        for (int kw = 0; kw < K_w; ++kw) {
            // Calculate input position that would contribute to this output
            int h_in = h_out + pad_h - kh * dilation_h;
            int w_in = w_out + pad_w - kw * dilation_w;
            
            // Check if the input position is valid after accounting for stride
            if (h_in >= 0 && h_in < H_in * stride_h && h_in % stride_h == 0 &&
                w_in >= 0 && w_in < W_in * stride_w && w_in % stride_w == 0) {
                
                h_in /= stride_h;
                w_in /= stride_w;
                
                if (h_in < H_in && w_in < W_in) {
                    // Calculate input channel index within the group
                    for (int c_in_group = 0; c_in_group < C_in / groups; ++c_in_group) {
                        int c_in = group_id * (C_in / groups) + c_in_group;
                        
                        int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                        int weight_idx = ((group_id * (C_out / groups) + c_out_in_group) * (C_in / groups) + c_in_group) * K_h * K_w + kh * K_w + kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    if (conv_bias != nullptr) {
        sum += conv_bias[c_out];
    }
    
    output[out_idx] = sum;
}

// ----------------------------------------------------------------------
// Fused kernel: subtract bias and apply tanh
// ----------------------------------------------------------------------
__global__ void fused_bias_tanh_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    const int N, const int C_out, const int H_out, const int W_out) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * H_out * W_out;
    
    if (idx >= total_elements) return;
    
    int remaining = idx % (C_out * H_out * W_out);
    int c_out = remaining / (H_out * W_out);
    
    float v = output[idx];
    v -= bias[c_out];
    v = tanhf(v);
    output[idx] = v;
}

// ----------------------------------------------------------------------
// Main entry point – fused transposed convolution + bias + tanh
// ----------------------------------------------------------------------
torch::Tensor fused_conv_transpose_tanh(
    torch::Tensor input,          // (N, C_in, H_in, W_in)
    torch::Tensor weight,         // (C_out, C_in/groups, K_h, K_w)
    torch::Tensor conv_bias,      // optional bias added after conv (size C_out) or empty
    int stride_h, int stride_w,
    int pad_h,   int pad_w,
    int out_pad_h, int out_pad_w,
    int dilation_h, int dilation_w,
    int groups,
    torch::Tensor bias)           // bias to subtract (size C_out)
{
    // Make sure tensors are contiguous and live on the GPU
    input   = input.contiguous();
    weight  = weight.contiguous();
    if (conv_bias.defined()) conv_bias = conv_bias.contiguous();
    bias    = bias.contiguous();

    const int N      = input.size(0);
    const int C_in   = input.size(1);
    const int H_in   = input.size(2);
    const int W_in   = input.size(3);

    const int C_out  = weight.size(0);
    const int K_h    = weight.size(2);
    const int K_w    = weight.size(3);

    // Output size (same formula as PyTorch's conv_transpose2d)
    const int H_out = (H_in - 1) * stride_h - 2 * pad_h +
                      dilation_h * (K_h - 1) + out_pad_h + 1;
    const int W_out = (W_in - 1) * stride_w - 2 * pad_w +
                      dilation_w * (K_w - 1) + out_pad_w + 1;

    // Allocate output tensor
    auto output = torch::empty({N, C_out, H_out, W_out}, input.options());

    // Raw device pointers
    const float* input_ptr   = input.data_ptr<float>();
    const float* weight_ptr  = weight.data_ptr<float>();
    float* output_ptr        = output.data_ptr<float>();
    
    const float* conv_bias_ptr = conv_bias.defined() && conv_bias.numel() > 0 ? 
                                conv_bias.data_ptr<float>() : nullptr;
    const float* bias_ptr    = bias.data_ptr<float>();

    // Launch transposed convolution kernel
    const int conv_total   = N * C_out * H_out * W_out;
    const int conv_threads = 256;
    const int conv_blocks  = (conv_total + conv_threads - 1) / conv_threads;
    
    conv_transpose2d_kernel<<<conv_blocks, conv_threads>>>(
        input_ptr, weight_ptr, conv_bias_ptr, output_ptr,
        N, C_in, H_in, W_in, C_out, K_h, K_w, H_out, W_out,
        stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w,
        dilation_h, dilation_w, groups);

    // Launch fused bias subtraction + tanh kernel
    const int fuse_total   = N * C_out * H_out * W_out;
    const int fuse_threads = 256;
    const int fuse_blocks  = (fuse_total + fuse_threads - 1) / fuse_threads;
    
    fused_bias_tanh_kernel<<<fuse_blocks, fuse_threads>>>(
        output_ptr, bias_ptr, N, C_out, H_out, W_out);
        
    cudaDeviceSynchronize();

    return output;
}
"""

# ----------------------------------------------------------------------
# C++ binding (PYBIND11) – exposes the CUDA kernel to Python
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

torch::Tensor fused_conv_transpose_tanh(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    int stride_h, int stride_w,
    int pad_h,   int pad_w,
    int out_pad_h, int out_pad_w,
    int dilation_h, int dilation_w,
    int groups,
    torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_tanh", &fused_conv_transpose_tanh,
          "Fused transposed convolution + bias + tanh");
}
"""

# ----------------------------------------------------------------------
# Build the inline extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Helper to turn a scalar or (h,w) tuple into a pair of ints
# ----------------------------------------------------------------------
def _to_pair(v):
    if isinstance(v, int):
        return (v, v)
    # assume it is a sequence of length 1 or 2
    if len(v) == 1:
        return (v[0], v[0])
    return (v[0], v[1])

# ----------------------------------------------------------------------
# The function that will be imported and evaluated
# ----------------------------------------------------------------------
def functional_model(
    x,                                 # input tensor (N,C,H,W)
    *,
    conv_transpose_weight,            # weight tensor
    conv_transpose_bias,              # optional bias added after conv
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    bias,                             # bias to subtract after conv
):
    # Move tensors to GPU and make contiguous
    x = x.cuda().contiguous()
    conv_transpose_weight = conv_transpose_weight.cuda().contiguous()
    bias = bias.cuda().contiguous()

    # If the user did not provide a conv_transpose_bias, pass an empty tensor
    if conv_transpose_bias is None:
        conv_bias = torch.empty(0, dtype=torch.float32, device='cuda')
    else:
        conv_bias = conv_transpose_bias.cuda().contiguous()

    # Unpack stride, padding, output_padding, dilation into (height, width)
    stride = _to_pair(conv_transpose_stride)
    pad    = _to_pair(conv_transpose_padding)
    out_pad = _to_pair(conv_transpose_output_padding)
    dilation = _to_pair(conv_transpose_dilation)

    groups = conv_transpose_groups if conv_transpose_groups is not None else 1

    # Call the fused CUDA kernel
    out = fused_ext.fused_conv_transpose_tanh(
        x,
        conv_transpose_weight,
        conv_bias,
        stride[0], stride[1],
        pad[0],    pad[1],
        out_pad[0], out_pad[1],
        dilation[0], dilation[1],
        groups,
        bias,
    )
    return out

# ----------------------------------------------------------------------
# Boilerplate required by the harness (kept unchanged)
# ----------------------------------------------------------------------
batch_size = 32
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
