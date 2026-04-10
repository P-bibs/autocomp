# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_072125/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv1d_weight', 'conv1d_bias', 'conv1d_stride', 'conv1d_padding', 'conv1d_dilation', 'conv1d_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv1d_weight', 'conv1d_bias']


class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool=False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

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
    # State for conv1d (nn.Conv2d)
    if 'conv1d_weight' in flat_state:
        state_kwargs['conv1d_weight'] = flat_state['conv1d_weight']
    else:
        state_kwargs['conv1d_weight'] = getattr(model.conv1d, 'weight', None)
    if 'conv1d_bias' in flat_state:
        state_kwargs['conv1d_bias'] = flat_state['conv1d_bias']
    else:
        state_kwargs['conv1d_bias'] = getattr(model.conv1d, 'bias', None)
    state_kwargs['conv1d_stride'] = model.conv1d.stride
    state_kwargs['conv1d_padding'] = model.conv1d.padding
    state_kwargs['conv1d_dilation'] = model.conv1d.dilation
    state_kwargs['conv1d_groups'] = model.conv1d.groups
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

# -------------------------------------------------------------------------
# CUDA kernel (device code) – optimized 2‑D convolution with tiling
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int k_h,
    int k_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int out_h,
    int out_w,
    int groups
) {
    // Shared memory for tiles
    __shared__ float sh_input[TILE_M][TILE_K];
    __shared__ float sh_weight[TILE_K][TILE_N];
    
    int batch_idx = blockIdx.z / out_channels;
    int out_ch = blockIdx.z % out_channels;
    
    int ho = blockIdx.y * TILE_M + threadIdx.y;
    int wo = blockIdx.x * TILE_N + threadIdx.x;
    
    if (ho >= out_h || wo >= out_w || batch_idx >= batch_size) return;
    
    // Group handling
    int group_id = out_ch * groups / out_channels;
    int in_ch_per_group = in_channels / groups;
    int out_ch_per_group = out_channels / groups;
    int weight_out_ch_offset = out_ch % out_ch_per_group;
    
    float sum = 0.0f;
    
    // Calculate base input position
    int base_h = ho * stride_h - pad_h;
    int base_w = wo * stride_w - pad_w;
    
    // Iterate through all input channels in the group
    for (int ic_start = 0; ic_start < in_ch_per_group; ic_start += TILE_K) {
        int tile_k_size = min(TILE_K, in_ch_per_group - ic_start);
        
        // Load input tile
        for (int kh = 0; kh < k_h; ++kh) {
            for (int kw = 0; kw < k_w; ++kw) {
                for (int ic = 0; ic < tile_k_size; ++ic) {
                    int in_ch_idx = group_id * in_ch_per_group + ic_start + ic;
                    int in_h = base_h + kh * dilation_h;
                    int in_w = base_w + kw * dilation_w;
                    
                    float val = 0.0f;
                    if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                        val = __ldg(&input[batch_idx * in_channels * height * width +
                                          in_ch_idx * height * width +
                                          in_h * width + in_w]);
                    }
                    if (threadIdx.y < tile_k_size && threadIdx.x == 0) {
                        sh_input[threadIdx.y][ic] = val;
                    }
                }
            }
        }
        
        __syncthreads();
        
        // Load weight tile
        for (int kh = 0; kh < k_h; ++kh) {
            for (int kw = 0; kw < k_w; ++kw) {
                for (int ic = 0; ic < tile_k_size; ++ic) {
                    int in_ch_idx = group_id * in_ch_per_group + ic_start + ic;
                    int weight_idx = weight_out_ch_offset * in_ch_per_group * k_h * k_w +
                                     (ic_start + ic) * k_h * k_w +
                                     kh * k_w + kw;
                    
                    float val = __ldg(&weight[weight_idx]);
                    if (threadIdx.y == 0 && threadIdx.x < tile_k_size) {
                        sh_weight[ic][threadIdx.x] = val;
                    }
                }
            }
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < k_h * k_w * tile_k_size; ++k) {
            int k_idx = k / (k_h * k_w);
            int remaining = k % (k_h * k_w);
            int kh = remaining / k_w;
            int kw = remaining % k_w;
            
            if (k_idx < tile_k_size) {
                sum += sh_input[threadIdx.y][k_idx] * sh_weight[k_idx][threadIdx.x];
            }
        }
        
        __syncthreads();
    }
    
    // Add bias if present
    if (bias != nullptr) {
        sum += __ldg(&bias[out_ch]);
    }
    
    // Write output
    output[batch_idx * out_channels * out_h * out_w +
           out_ch * out_h * out_w +
           ho * out_w + wo] = sum;
}

// Host wrapper
void conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int k_h,
    int k_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int out_h,
    int out_w,
    int groups
) {
    dim3 grid((out_w + TILE_N - 1) / TILE_N,
              (out_h + TILE_M - 1) / TILE_M,
              batch_size * out_channels);
    dim3 block(TILE_N, TILE_M, 1);
    
    conv2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        height, width,
        k_h, k_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        out_h, out_w,
        groups
    );
    
    cudaDeviceSynchronize();
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int k_h,
    int k_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int out_h,
    int out_w,
    int groups
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_forward", &conv2d_forward, "Optimized 2D convolution forward pass");
}
"""

# -------------------------------------------------------------------------
# Build the inline extension with aggressive optimisation flags
# -------------------------------------------------------------------------
conv_ext = load_inline(
    name='conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Helper: compute output spatial size
# -------------------------------------------------------------------------
def compute_output_size(H, W, kH, kW, stride, padding, dilation):
    if isinstance(stride, int):
        sH = sW = stride
    else:
        sH, sW = stride
    if isinstance(padding, int):
        pH = pW = padding
    else:
        pH, pW = padding
    if isinstance(dilation, int):
        dH = dW = dilation
    else:
        dH, dW = dilation
    out_h = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    out_w = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
    return out_h, out_w

# -------------------------------------------------------------------------
# functional_model – the only function that will be imported
# -------------------------------------------------------------------------
def functional_model(
    x,
    *,
    conv1d_weight,
    conv1d_bias,
    conv1d_stride,
    conv1d_padding,
    conv1d_dilation,
    conv1d_groups,
):
    # Move tensors to GPU
    x = x.cuda()
    conv1d_weight = conv1d_weight.cuda()
    if conv1d_bias is not None:
        conv1d_bias = conv1d_bias.cuda()
    
    # Extract dimensions
    N, C_in, H, W = x.shape
    out_ch, in_ch_per_group, kH, kW = conv1d_weight.shape
    C_out = out_ch
    C_in = in_ch_per_group * conv1d_groups
    
    # Compute output spatial dimensions
    out_h, out_w = compute_output_size(
        H, W, kH, kW,
        conv1d_stride, conv1d_padding, conv1d_dilation
    )
    
    # Allocate output tensor
    output = torch.empty((N, C_out, out_h, out_w), dtype=x.dtype, device='cuda')
    
    # Normalize parameters to tuples
    if isinstance(conv1d_stride, int):
        stride_h = stride_w = conv1d_stride
    else:
        stride_h, stride_w = conv1d_stride
        
    if isinstance(conv1d_padding, int):
        pad_h = pad_w = conv1d_padding
    else:
        pad_h, pad_w = conv1d_padding
        
    if isinstance(conv1d_dilation, int):
        dilation_h = dilation_w = conv1d_dilation
    else:
        dilation_h, dilation_w = conv1d_dilation
    
    # Launch kernel
    conv_ext.conv2d_forward(
        x, conv1d_weight, conv1d_bias, output,
        N, C_in, C_out, H, W,
        kH, kW,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        out_h, out_w,
        conv1d_groups
    )
    
    return output
