# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_100757/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a 3D convolution, applies minimum operation along a specific dimension, 
    and then applies softmax.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

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
    # State for conv (nn.Conv3d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
    if 'dim' in flat_state:
        state_kwargs['dim'] = flat_state['dim']
    else:
        state_kwargs['dim'] = getattr(model, 'dim')
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

# Optimized CUDA kernel implementing fused 3D convolution + min reduction
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__global__ void fused_conv_min_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int D, int H, int W,
    int C_out, int K,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int dim
) {
    // Each block handles one output spatial location (d, h, w) for all output channels of one sample
    int b = blockIdx.x;
    int out_d_idx = blockIdx.y;
    int out_h_idx = blockIdx.z;
    int out_w_idx = threadIdx.x;

    if (b >= B || out_w_idx >= (W + 2 * pad_w - dilation_w * (K - 1) - 1) / stride_w + 1) return;

    // Shared memory for partial reductions across channels
    extern __shared__ float sdata[];

    int out_d = out_d_idx * stride_d - pad_d;
    int out_h = out_h_idx * stride_h - pad_h;
    int out_w = out_w_idx * stride_w - pad_w;

    float min_vals[8]; // Process up to 8 output channels per thread for better occupancy

    // Initialize with large value
    for (int co = 0; co < 8 && threadIdx.x * 8 + co < C_out; co++) {
        min_vals[co] = 1e10f;
    }

    // Loop over input channels (C_in)
    for (int ci = 0; ci < C_in; ci++) {
        // Loop over kernel positions
        for (int kd = 0; kd < K; kd++) {
            int d = out_d + kd * dilation_d;
            if (d < 0 || d >= D) continue;
            
            for (int kh = 0; kh < K; kh++) {
                int h = out_h + kh * dilation_h;
                if (h < 0 || h >= H) continue;
                
                for (int kw = 0; kw < K; kw++) {
                    int w = out_w + kw * dilation_w;
                    if (w < 0 || w >= W) continue;

                    float input_val = input[(((b * C_in + ci) * D + d) * H + h) * W + w];

                    // Compute for multiple output channels per thread
                    for (int co = 0; co < 8 && threadIdx.x * 8 + co < C_out; co++) {
                        int actual_co = threadIdx.x * 8 + co;
                        float weight_val = weight[(((actual_co * C_in + ci) * K + kd) * K + kh) * K + kw];
                        float conv_val = input_val * weight_val;
                        if (ci == 0 && kd == 0 && kh == 0 && kw == 0) {
                            min_vals[co] = conv_val;
                        } else {
                            min_vals[co] += conv_val;
                        }
                    }
                }
            }
        }
    }

    // Add bias
    for (int co = 0; co < 8 && threadIdx.x * 8 + co < C_out; co++) {
        int actual_co = threadIdx.x * 8 + co;
        min_vals[co] += bias[actual_co];
    }

    // Reduce along specified dimension
    if (dim == 2) { // Depth dimension
        // Store in shared memory for reduction
        for (int co = 0; co < 8 && threadIdx.x * 8 + co < C_out; co++) {
            sdata[threadIdx.x * 8 + co] = min_vals[co];
        }
        __syncthreads();

        // Reduction in shared memory
        for (int stride = 1; stride < ((W + 2 * pad_w - dilation_w * (K - 1) - 1) / stride_w + 1 + 1) / 2; stride *= 2) {
            int index = (threadIdx.x + stride) * 8;
            if (index < C_out && (threadIdx.x * 8) < C_out) {
                for (int co = 0; co < 8 && (threadIdx.x * 8 + co) < C_out; co++) {
                    sdata[threadIdx.x * 8 + co] = fminf(sdata[threadIdx.x * 8 + co], sdata[(threadIdx.x + stride) * 8 + co]);
                }
            }
            __syncthreads();
        }

        // Write result
        if (threadIdx.x == 0) {
            for (int co = 0; co < 8 && co < C_out; co++) {
                output[((b * C_out + co) * ((H + 2 * pad_h - dilation_h * (K - 1) - 1) / stride_h + 1) + out_h_idx) * 
                       ((W + 2 * pad_w - dilation_w * (K - 1) - 1) / stride_w + 1) + out_w_idx] = sdata[co];
            }
        }
    } else {
        // For other dimensions, write directly
        for (int co = 0; co < 8 && threadIdx.x * 8 + co < C_out; co++) {
            int actual_co = threadIdx.x * 8 + co;
            int out_d_size = (D + 2 * pad_d - dilation_d * (K - 1) - 1) / stride_d + 1;
            int out_h_size = (H + 2 * pad_h - dilation_h * (K - 1) - 1) / stride_h + 1;
            int out_w_size = (W + 2 * pad_w - dilation_w * (K - 1) - 1) / stride_w + 1;

            int out_idx;
            if (dim == 3) { // Height dimension
                out_idx = ((b * C_out + actual_co) * out_h_size + out_h_idx) * out_w_size + out_w_idx;
            } else if (dim == 4) { // Width dimension
                out_idx = ((b * C_out + actual_co) * out_h_size + out_h_idx) * out_w_size + out_w_idx;
            } else {
                out_idx = ((b * C_out + actual_co) * out_d_size + out_d_idx) * out_h_size * out_w_size + 
                          out_h_idx * out_w_size + out_w_idx;
            }
            output[out_idx] = min_vals[co];
        }
    }
}

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int dim
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    const int B = input.size(0);
    const int C_in = input.size(1);
    const int D = input.size(2);
    const int H = input.size(3);
    const int W = input.size(4);
    const int C_out = weight.size(0);
    const int K = weight.size(2);

    // Calculate output dimensions
    const int out_d = (D + 2 * pad_d - dilation_d * (K - 1) - 1) / stride_d + 1;
    const int out_h = (H + 2 * pad_h - dilation_h * (K - 1) - 1) / stride_h + 1;
    const int out_w = (W + 2 * pad_w - dilation_w * (K - 1) - 1) / stride_w + 1;

    dim3 threads(min(256, out_w));
    dim3 blocks(B, out_d, out_h);

    // Shared memory size for reduction
    int shared_mem_size = C_out * sizeof(float);

    fused_conv_min_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C_in, D, H, W,
        C_out, K,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w,
        dim
    );
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int dim
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused 3D Convolution and Min Reduction");
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
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
    dim,
):
    # Ensure inputs are contiguous
    x = x.contiguous()
    conv_weight = conv_weight.contiguous()
    conv_bias = conv_bias.contiguous()
    
    # Handle stride, padding, dilation
    if isinstance(conv_stride, int):
        stride_d = stride_h = stride_w = conv_stride
    else:
        stride_d, stride_h, stride_w = conv_stride
    
    if isinstance(conv_padding, int):
        pad_d = pad_h = pad_w = conv_padding
    else:
        pad_d, pad_h, pad_w = conv_padding
        
    if isinstance(conv_dilation, int):
        dilation_d = dilation_h = dilation_w = conv_dilation
    else:
        dilation_d, dilation_h, dilation_w = conv_dilation
    
    # Adjust dim for CUDA kernel (account for batch dimension)
    cuda_dim = dim + 2  # [B, C, D, H, W] -> dim + 2
    
    # Calculate output shape
    B, C_in, D, H, W = x.shape
    C_out = conv_weight.shape[0]
    K = conv_weight.shape[2]
    
    out_d = (D + 2 * pad_d - dilation_d * (K - 1) - 1) // stride_d + 1
    out_h = (H + 2 * pad_h - dilation_h * (K - 1) - 1) // stride_h + 1
    out_w = (W + 2 * pad_w - dilation_w * (K - 1) - 1) // stride_w + 1
    
    # Determine output shape after reduction
    out_shape = [B, C_out]
    if cuda_dim != 2:  # Not reducing depth
        out_shape.append(out_d)
    if cuda_dim != 3:  # Not reducing height
        out_shape.append(out_h)
    if cuda_dim != 4:  # Not reducing width
        out_shape.append(out_w)
    
    # Create output tensor
    res = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_op(
        x, conv_weight, conv_bias, res,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w,
        cuda_dim
    )
    
    # Final softmax
    return torch.softmax(res, dim=1)

# Test parameters
batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
dim = 2

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]
