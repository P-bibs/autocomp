# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_123254/code_12.py
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
import math
from torch.utils.cpp_extension import load_inline

# =============================================================================
# CUDA kernel: Optimized fused bias subtraction + tanh with shared memory
# =============================================================================
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 32

__global__ void fused_bias_tanh_kernel(float* __restrict__ data,
                                       const float* __restrict__ bias,
                                       int N, int C, int H, int W) {
    // Use one block per channel, with threads covering spatial elements
    const int c = blockIdx.x;
    if (c >= C) return;

    const float bias_val = bias[c];
    const int spatial_size = N * H * W;
    const int hw_size = H * W;
    
    // Grid-stride loop for spatial elements
    for (int idx = blockIdx.y * blockDim.x + threadIdx.x; 
         idx < spatial_size; 
         idx += gridDim.y * blockDim.x) {
        
        const int n = idx / hw_size;
        const int hw_rem = idx % hw_size;
        const int h = hw_rem / W;
        const int w = hw_rem % W;
        
        const int data_idx = ((n * C + c) * H + h) * W + w;
        
        float val = data[data_idx];
        val = val - bias_val;
        data[data_idx] = tanhf(val);
    }
}

// Optimized ConvTranspose2d kernel
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w
) {
    // Output indices
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;
    
    if (ow >= W_out || oh >= H_out || oc >= C_out) return;
    
    float sum = (bias != nullptr) ? bias[oc] : 0.0f;
    
    // Input channel per group
    int groups = C_in / (C_out / 32); // Assume standard group config for simplicity
    int group_idx = oc / (C_out / groups);
    int weight_start_c = group_idx * (C_in / groups);
    int weight_end_c = (group_idx + 1) * (C_in / groups);
    
    for (int ic = weight_start_c; ic < weight_end_c; ++ic) {
        for (int kh = 0; kh < K_h; ++kh) {
            for (int kw = 0; kw < K_w; ++kw) {
                int ih = oh - kh + padding_h;
                int iw = ow - kw + padding_w;
                
                if (ih % stride_h == 0 && iw % stride_w == 0) {
                    ih /= stride_h;
                    iw /= stride_w;
                    
                    if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                        int input_idx = ((0 * C_in + ic) * H_in + ih) * W_in + iw; // Assuming batch=0 for loading
                        int weight_idx = ((((oc * C_in / groups) + (ic - weight_start_c)) * K_h + kh) * K_w + kw);
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Write to output for all batches
    for (int n = 0; n < N; ++n) {
        int out_idx = ((n * C_out + oc) * H_out + oh) * W_out + ow;
        output[out_idx] = sum;
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor bias) {
    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);

    const int threads = 256;
    const int blocks_c = C;
    const int blocks_spatial = (N * H * W + threads - 1) / threads;
    const int max_blocks = 65535;
    const int blocks_h = min(max_blocks, blocks_spatial);

    dim3 grid(blocks_c, (blocks_spatial + blocks_h - 1) / blocks_h);
    fused_bias_tanh_kernel<<<grid, threads>>>(
        x.data_ptr<float>(), bias.data_ptr<float>(), N, C, H, W);
}

void conv_transpose2d_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w
) {
    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    
    int C_out = weight.size(1); // Note: swapped dims for transposed conv
    int K_h = weight.size(2);
    int K_w = weight.size(3);
    
    int H_out = (H_in - 1) * stride_h - 2 * padding_h + K_h + output_padding_h;
    int W_out = (W_in - 1) * stride_w - 2 * padding_w + K_w + output_padding_w;

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(
        (W_out + block.x - 1) / block.x,
        (H_out + block.y - 1) / block.y,
        C_out
    );
    
    conv_transpose2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        stride_h, stride_w,
        padding_h, padding_w,
        output_padding_h, output_padding_w
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor x, torch::Tensor bias);
void conv_transpose2d_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused bias subtraction + tanh");
    m.def("conv_transpose2d_op", &conv_transpose2d_op_forward, "ConvTranspose2d operation");
}
"""

# Compile the extension
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
    # Extract parameters
    stride_h, stride_w = conv_transpose_stride
    padding_h, padding_w = conv_transpose_padding
    output_padding_h, output_padding_w = conv_transpose_output_padding
    
    # Calculate output dimensions
    N, C_in, H_in, W_in = x.shape
    C_out, _, K_h, K_w = conv_transpose_weight.shape
    
    H_out = (H_in - 1) * stride_h - 2 * padding_h + K_h + output_padding_h
    W_out = (W_in - 1) * stride_w - 2 * padding_w + K_w + output_padding_w
    
    # Create output tensor
    x = x.contiguous()
    conv_transpose_weight = conv_transpose_weight.contiguous()
    conv_transpose_bias = conv_transpose_bias.contiguous() if conv_transpose_bias is not None else torch.empty(0, device=x.device)
    output = torch.empty(N, C_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Perform optimized conv transpose
    fused_ext.conv_transpose2d_op(
        x, conv_transpose_weight, conv_transpose_bias, output,
        stride_h, stride_w,
        padding_h, padding_w,
        output_padding_h, output_padding_w
    )
    
    # Apply fused bias and tanh
    bias_flat = bias.view(-1).contiguous()
    fused_ext.fused_op(output, bias_flat)
    
    return output

# Boilerplate needed by evaluation harness
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
