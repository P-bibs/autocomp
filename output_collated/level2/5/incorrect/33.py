# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_115905/code_5.py
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

# --- CUDA Kernel ---
# We perform the full operation in a single kernel: conv_transpose + bias subtraction + tanh
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

// Helper to compute linear index from 5D coordinates (for col2im buffer)
__device__ int get_idx_5d(int n, int c, int h, int w, int d, int C, int H, int W, int D) {
    return ((n * C + c) * H + h) * W * D + w * D + d;
}

// Helper to compute linear index from 4D coordinates
__device__ int get_idx_4d(int n, int c, int h, int w, int C, int H, int W) {
    return ((n * C + c) * H + h) * W + w;
}

// Helper to compute linear index from 6D (for GEMM output)
__device__ int get_idx_6d(int n, int co, int kx, int ky, int ix, int iy, 
                          int Co, int Kx, int Ky, int Ix, int Iy) {
    return ((((n * Co + co) * Kx + kx) * Ky + ky) * Ix + ix) * Iy + iy;
}

__global__ void im2col_kernel(
    const float* input, float* col_buffer,
    int N, int C, int H, int W,
    int Kx, int Ky,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dil_h, int dil_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = N * C * Kx * Ky * H * W;
    
    if (idx >= total_threads) return;
    
    int tmp = idx;
    int w_out = tmp % W; tmp /= W;
    int h_out = tmp % H; tmp /= H;
    int ky = tmp % Ky; tmp /= Ky;
    int kx = tmp % Kx; tmp /= Kx;
    int c = tmp % C; tmp /= C;
    int n = tmp;
    
    int h_in = h_out * stride_h - pad_h + kx * dil_h;
    int w_in = w_out * stride_w - pad_w + ky * dil_w;
    
    if (h_in >= 0 && h_in < (H * stride_h - 2 * pad_h + (Kx - 1) * dil_h + 1) &&
        w_in >= 0 && w_in < (W * stride_w - 2 * pad_w + (Ky - 1) * dil_w + 1)) {
        float val = 0.0f;
        if (h_in < H && w_in < W && h_in >= 0 && w_in >= 0) {
            val = input[get_idx_4d(n, c, h_in, w_in, C, H, W)];
        }
        col_buffer[idx] = val;
    } else {
        col_buffer[idx] = 0.0f;
    }
}

__global__ void fused_conv_transpose_tanh_kernel(
    const float* input, const float* weight, const float* conv_bias_data, const float* bias_data,
    float* output,
    int N, int Co, int Ho, int Wo,
    int Ci, int Hi, int Wi,
    int Kx, int Ky,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_pad_h, int out_pad_w,
    int dil_h, int dil_w,
    int groups
) {
    // Each thread computes one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = N * Co * Ho * Wo;
    
    if (idx >= total_outputs) return;
    
    int tmp = idx;
    int w_out = tmp % Wo; tmp /= Wo;
    int h_out = tmp % Ho; tmp /= Ho;
    int co = tmp % Co; tmp /= Co;
    int n = tmp;
    
    int group_id = co * groups / Co;
    int co_in_group = co - group_id * (Co / groups);
    
    float sum = 0.0f;
    
    // Loop over all input positions that can influence this output
    for (int kx = 0; kx < Kx; kx++) {
        for (int ky = 0; ky < Ky; ky++) {
            // Compute input position that contributes to this output
            int h_in = h_out - kx * dil_h + pad_h;
            int w_in = w_out - ky * dil_w + pad_w;
            
            if (h_in % stride_h == 0 && w_in % stride_w == 0) {
                h_in /= stride_h;
                w_in /= stride_w;
                
                if (h_in >= 0 && h_in < Hi && w_in >= 0 && w_in < Wi) {
                    // Weight tensor is [Ci, Co/groups, Kx, Ky] in PyTorch
                    int weight_idx = ((group_id * (Co/groups) + co_in_group) * Ci + 0) * Kx * Ky + 
                                     (Kx - 1 - kx) * Ky + (Ky - 1 - ky);
                    int input_idx = get_idx_4d(n, 0, h_in, w_in, Ci, Hi, Wi);
                    
                    for (int ci = 0; ci < Ci; ci++) {
                        float weight_val = weight[weight_idx + ci * Kx * Ky];
                        float input_val = input[input_idx + ci * Hi * Wi];
                        sum += weight_val * input_val;
                    }
                }
            }
        }
    }
    
    // Add convolution bias if present
    if (conv_bias_data != nullptr) {
        sum += conv_bias_data[co];
    }
    
    // Subtract our custom bias
    float bias_val = bias_data[co];
    sum -= bias_val;
    
    // Apply tanh activation
    output[idx] = tanhf(sum);
}

void fused_conv_transpose_tanh_op(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_pad_h, int out_pad_w,
    int dil_h, int dil_w,
    int groups
) {
    int N = input.size(0);
    int Ci = input.size(1);
    int Hi = input.size(2);
    int Wi = input.size(3);
    
    int Co = weight.size(1) * groups; // weight is [Ci, Co/groups, Kx, Ky]
    int Kx = weight.size(2);
    int Ky = weight.size(3);
    
    int Ho = (Hi - 1) * stride_h - 2 * pad_h + Kx + out_pad_h;
    int Wo = (Wi - 1) * stride_w - 2 * pad_w + Ky + out_pad_w;
    
    int total_outputs = N * Co * Ho * Wo;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    
    const float* conv_bias_ptr = conv_bias.defined() ? conv_bias.data_ptr<float>() : nullptr;
    
    fused_conv_transpose_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias_ptr,
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, Co, Ho, Wo,
        Ci, Hi, Wi,
        Kx, Ky,
        stride_h, stride_w,
        pad_h, pad_w,
        out_pad_h, out_pad_w,
        dil_h, dil_w,
        groups
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_tanh_op(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_pad_h, int out_pad_w,
    int dil_h, int dil_w,
    int groups
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_tanh", &fused_conv_transpose_tanh_op, "Fused ConvTranspose2d + Bias Sub + Tanh");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose_tanh_ext',
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
    # Extract dimensions
    N, Ci, Hi, Wi = x.shape
    Co = conv_transpose_weight.shape[1] * conv_transpose_groups
    Kx, Ky = conv_transpose_weight.shape[2], conv_transpose_weight.shape[3]
    
    # Handle stride, padding, dilation tuples
    if isinstance(conv_transpose_stride, int):
        stride_h = stride_w = conv_transpose_stride
    else:
        stride_h, stride_w = conv_transpose_stride
    
    if isinstance(conv_transpose_padding, int):
        pad_h = pad_w = conv_transpose_padding
    else:
        pad_h, pad_w = conv_transpose_padding
        
    if isinstance(conv_transpose_output_padding, int):
        out_pad_h = out_pad_w = conv_transpose_output_padding
    else:
        out_pad_h, out_pad_w = conv_transpose_output_padding
        
    if isinstance(conv_transpose_dilation, int):
        dil_h = dil_w = conv_transpose_dilation
    else:
        dil_h, dil_w = conv_transpose_dilation

    # Compute output dimensions
    Ho = (Hi - 1) * stride_h - 2 * pad_h + Kx + out_pad_h
    Wo = (Wi - 1) * stride_w - 2 * pad_w + Ky + out_pad_w
    
    # Create output tensor
    output = torch.empty((N, Co, Ho, Wo), device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_conv_transpose_tanh(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias if conv_transpose_bias is not None else torch.tensor([], device=x.device),
        bias.view(-1),
        output,
        stride_h, stride_w,
        pad_h, pad_w,
        out_pad_h, out_pad_w,
        dil_h, dil_w,
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
