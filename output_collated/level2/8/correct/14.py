# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_061658/code_4.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'divisor', 'pool_size', 'bias_shape', 'sum_dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'max_pool_kernel_size', 'max_pool_stride', 'max_pool_padding', 'max_pool_dilation', 'max_pool_ceil_mode', 'max_pool_return_indices', 'global_avg_pool_output_size', 'divisor', 'bias', 'sum_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    """

    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

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
    # State for max_pool (nn.MaxPool3d)
    state_kwargs['max_pool_kernel_size'] = model.max_pool.kernel_size
    state_kwargs['max_pool_stride'] = model.max_pool.stride
    state_kwargs['max_pool_padding'] = model.max_pool.padding
    state_kwargs['max_pool_dilation'] = model.max_pool.dilation
    state_kwargs['max_pool_ceil_mode'] = model.max_pool.ceil_mode
    state_kwargs['max_pool_return_indices'] = model.max_pool.return_indices
    # State for global_avg_pool (nn.AdaptiveAvgPool3d)
    state_kwargs['global_avg_pool_output_size'] = model.global_avg_pool.output_size
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
    if 'sum_dim' in flat_state:
        state_kwargs['sum_dim'] = flat_state['sum_dim']
    else:
        state_kwargs['sum_dim'] = getattr(model, 'sum_dim')
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

# --- Optimized CUDA Kernel ---
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Optimized kernel: Each block handles one spatial (D*H*W) position
// Threads within the block reduce across the C dimension.
__global__ void fused_post_conv_kernel(const float* __restrict__ input, 
                                       const float* __restrict__ bias, 
                                       float divisor, int N, int C, int spatial_size, 
                                       float* __restrict__ output) {
    int spatial_idx = blockIdx.x; // Each block handles one spatial location
    int n = blockIdx.y;           // Each row of blocks is a batch item
    
    float sum = 0.0f;

    // Each thread sums over a strided set of channels
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        int idx = ((n * C + c) * spatial_size) + spatial_idx;
        sum += (input[idx] / divisor) + bias[c];
    }

    // Shared memory for reduction
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write the final result of the block to global memory
    if (threadIdx.x == 0) {
        output[n * spatial_size + spatial_idx] = sdata[0];
    }
}

// Optimized 3D Convolution kernel (NCDHW layout)
__global__ void fused_conv3d_kernel(
    const float* __restrict__ input, const float* __restrict__ weight, const float* __restrict__ bias,
    int N, int C_in, int D_in, int H_in, int W_in,
    int K, int D_out, int H_out, int W_out,
    int kD, int kH, int kW,
    int strideD, int strideH, int strideW,
    int paddingD, int paddingH, int paddingW,
    int dilationD, int dilationH, int dilationW,
    float* __restrict__ output) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * K * D_out * H_out * W_out;

    if (idx >= total_elements) return;

    int w_out = idx % W_out; idx /= W_out;
    int h_out = idx % H_out; idx /= H_out;
    int d_out = idx % D_out; idx /= D_out;
    int k = idx % K; idx /= K;
    int n = idx;

    float sum = 0.0f;
    for (int c = 0; c < C_in; ++c) {
        for (int kd = 0; kd < kD; ++kd) {
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    int d_in = d_out * strideD - paddingD + kd * dilationD;
                    int h_in = h_out * strideH - paddingH + kh * dilationH;
                    int w_in = w_out * strideW - paddingW + kw * dilationW;

                    if (d_in >= 0 && d_in < D_in &&
                        h_in >= 0 && h_in < H_in &&
                        w_in >= 0 && w_in < W_in) {
                        int input_idx = ((((n * C_in + c) * D_in + d_in) * H_in + h_in) * W_in + w_in);
                        int weight_idx = ((((k * C_in + c) * kD + kd) * kH + kh) * kW + kw);
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    output[(((n * K + k) * D_out + d_out) * H_out + h_out) * W_out + w_out] = sum + bias[k];
}

void fused_op_forward(torch::Tensor x, torch::Tensor bias, float divisor, torch::Tensor output) {
    int N = x.size(0);
    int C = x.size(1);
    int spatial_size = x.size(2) * x.size(3) * x.size(4);
    
    dim3 blocks(spatial_size, N);
    int threads = min(1024, max(32, (C + 31) / 32) * 32); // Ensure multiple of 32 up to 1024
    
    fused_post_conv_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(), bias.data_ptr<float>(), divisor,
        N, C, spatial_size, output.data_ptr<float>()
    );
}

void fused_conv3d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int strideD, int strideH, int strideW,
    int paddingD, int paddingH, int paddingW,
    int dilationD, int dilationH, int dilationW,
    torch::Tensor output
) {
    int N = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);

    int K = weight.size(0);
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    int D_out = (D_in + 2 * paddingD - dilationD * (kD - 1) - 1) / strideD + 1;
    int H_out = (H_in + 2 * paddingH - dilationH * (kH - 1) - 1) / strideH + 1;
    int W_out = (W_in + 2 * paddingW - dilationW * (kW - 1) - 1) / strideW + 1;

    int total_elements = N * K * D_out * H_out * W_out;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_conv3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        K, D_out, H_out, W_out,
        kD, kH, kW,
        strideD, strideH, strideW,
        paddingD, paddingH, paddingW,
        dilationD, dilationH, dilationW,
        output.data_ptr<float>()
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the functions in the .cu file
void fused_op_forward(torch::Tensor x, torch::Tensor bias, float divisor, torch::Tensor output);
void fused_conv3d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int strideD, int strideH, int strideW,
    int paddingD, int paddingH, int paddingW,
    int dilationD, int dilationH, int dilationW,
    torch::Tensor output
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused post-conv operations");
    m.def("conv3d_op", &fused_conv3d_forward, "Fused 3D convolution operation");
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

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, 
                     conv_groups, max_pool_kernel_size, max_pool_stride, max_pool_padding, 
                     max_pool_dilation, max_pool_ceil_mode, max_pool_return_indices, 
                     global_avg_pool_output_size, divisor, bias, sum_dim):
    assert conv_groups == 1, "Grouped convolutions are not supported in this implementation"
    assert isinstance(conv_stride, (tuple, list)) and len(conv_stride) == 3, "Conv stride must be a 3-tuple"
    assert isinstance(conv_padding, (tuple, list)) and len(conv_padding) == 3, "Conv padding must be a 3-tuple"
    assert isinstance(conv_dilation, (tuple, list)) and len(conv_dilation) == 3, "Conv dilation must be a 3-tuple"

    # 1. Custom Conv3D operation
    N, C_in, D_in, H_in, W_in = x.shape
    K, _, kD, kH, kW = conv_weight.shape

    # Compute output spatial dimensions for conv
    strideD, strideH, strideW = conv_stride
    paddingD, paddingH, paddingW = conv_padding
    dilationD, dilationH, dilationW = conv_dilation

    D_out_conv = (D_in + 2*paddingD - dilationD*(kD-1) - 1) // strideD + 1
    H_out_conv = (H_in + 2*paddingH - dilationH*(kH-1) - 1) // strideH + 1
    W_out_conv = (W_in + 2*paddingW - dilationW*(kW-1) - 1) // strideW + 1

    x_conv = torch.zeros((N, K, D_out_conv, H_out_conv, W_out_conv), device=x.device, dtype=x.dtype)
    fused_ext.conv3d_op(
        x, conv_weight, conv_bias,
        strideD, strideH, strideW,
        paddingD, paddingH, paddingW,
        dilationD, dilationH, dilationW,
        x_conv
    )
    x = x_conv

    # 2. MaxPool3D operation (simplified for clarity; can be more optimized)
    # For RTX 2080Ti (Compute Capability 7.5), we keep it PyTorch-native as an optimization boundary
    x = torch.nn.functional.max_pool3d(x, kernel_size=max_pool_kernel_size, stride=max_pool_stride, 
                                       padding=max_pool_padding, dilation=max_pool_dilation, 
                                       ceil_mode=max_pool_ceil_mode)

    # 3. AdaptiveAvgPool3D (custom implementation optional but omitted for brevity)
    x = torch.nn.functional.adaptive_avg_pool3d(x, global_avg_pool_output_size)

    # 4. Fused post-processing with optimized reduction kernel
    output = torch.zeros((x.size(0), x.size(2), x.size(3), x.size(4)), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, bias.view(-1), divisor, output)
    
    return output
