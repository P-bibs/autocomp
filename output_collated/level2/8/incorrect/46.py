# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_064410/code_1.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// Helper function to compute conv3d output size
__device__ inline int get_conv_output_size(int input_size, int pad, int dilation, int kernel, int stride) {
    return (input_size + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
}

// Helper function to compute maxpool3d output size
__device__ inline int get_pool_output_size(int input_size, int pad, int dilation, int kernel, int stride) {
    return (input_size + 2 * pad - dilation * (kernel - 1)) / stride + 1;
}

// Fused kernel performing the entire pipeline:
// Conv3D -> MaxPool3D -> AdaptiveAvgPool3D -> Channel Summation -> Bias/Div
__global__ void fused_pipeline_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight, 
    const float* __restrict__ conv_bias,
    int N, int C_in, int D, int H, int W,
    int K_D, int K_H, int K_W,                // conv kernel sizes
    int S_D, int S_H, int S_W,                // conv strides
    int P_D, int P_H, int P_W,                // conv padding
    int dilation_D, int dilation_H, int dilation_W, // conv dilation
    int D_out_conv, int H_out_conv, int W_out_conv,  // conv output dims
    
    int P_D_max, int P_H_max, int P_W_max,    // maxpool padding
    int K_D_max, int K_H_max, int K_W_max,    // maxpool kernel sizes
    int S_D_max, int S_H_max, int S_W_max,    // maxpool strides
    int dilation_D_max, int dilation_H_max, int dilation_W_max, // maxpool dilation
    int D_out_max, int H_out_max, int W_out_max,     // maxpool output dims
    
    int D_out_adapt, int H_out_adapt, int W_out_adapt, // adaptive pool output dims
    
    float divisor, float total_bias,
    float* __restrict__ output) 
{
    // Each thread block handles one output spatial location
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= N * D_out_adapt * H_out_adapt * W_out_adapt) return;
    
    int n = out_idx / (D_out_adapt * H_out_adapt * W_out_adapt);
    int tmp = out_idx % (D_out_adapt * H_out_adapt * W_out_adapt);
    int d_out_adapt = tmp / (H_out_adapt * W_out_adapt);
    int h_out_adapt = (tmp % (H_out_adapt * W_out_adapt)) / W_out_adapt;
    int w_out_adapt = tmp % W_out_adapt;
    
    // Determine the maxpool region for this adaptive pool output
    // Simplified: assume uniform division for adaptive pooling
    int d_start = (d_out_adapt * D_out_max) / D_out_adapt;
    int d_end = ((d_out_adapt + 1) * D_out_max + D_out_adapt - 1) / D_out_adapt;
    int h_start = (h_out_adapt * H_out_max) / H_out_adapt;
    int h_end = ((h_out_adapt + 1) * H_out_max + H_out_adapt - 1) / H_out_adapt;
    int w_start = (w_out_adapt * W_out_max) / W_out_adapt;
    int w_end = ((w_out_adapt + 1) * W_out_max + W_out_adapt - 1) / W_out_adapt;
    
    float sum = 0.0f;
    
    // For each output channel (assuming C_out=1 as in original)
    // In a more general case, this would be a loop over output channels
    int c_out = 0;
    float channel_max = -1e30f;
    
    // Iterate through maxpool region
    for (int d_max = d_start; d_max < d_end; d_max++) {
        for (int h_max = h_start; h_max < h_end; h_max++) {
            for (int w_max = w_start; w_max < w_end; w_max++) {
                // Compute conv3d values for this maxpool input position
                float conv_val = 0.0f;
                
                // Determine conv output position that feeds into maxpool
                int d_conv = d_max * S_D_max - P_D_max;
                int h_conv = h_max * S_H_max - P_H_max;
                int w_conv = w_max * S_W_max - P_W_max;
                
                // Check if this is within valid conv output range
                if (d_conv >= 0 && d_conv < D_out_conv &&
                    h_conv >= 0 && h_conv < H_out_conv &&
                    w_conv >= 0 && w_conv < W_out_conv) {
                    
                    // Perform conv3d computation
                    for (int c_in = 0; c_in < C_in; c_in++) {
                        for (int kd = 0; kd < K_D; kd++) {
                            for (int kh = 0; kh < K_H; kh++) {
                                for (int kw = 0; kw < K_W; kw++) {
                                    int d_in = d_conv * S_D - P_D + kd * dilation_D;
                                    int h_in = h_conv * S_H - P_H + kh * dilation_H;
                                    int w_in = w_conv * S_W - P_W + kw * dilation_W;
                                    
                                    if (d_in >= 0 && d_in < D && 
                                        h_in >= 0 && h_in < H && 
                                        w_in >= 0 && w_in < W) {
                                        int input_idx = ((n * C_in + c_in) * D + d_in) * H * W + h_in * W + w_in;
                                        int weight_idx = ((c_out * C_in + c_in) * K_D + kd) * K_H * K_W + kh * K_W + kw;
                                        conv_val += input[input_idx] * weight[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                    
                    // Add bias
                    conv_val += conv_bias[c_out];
                }
                
                // Update max value
                if (conv_val > channel_max) {
                    channel_max = conv_val;
                }
            }
        }
    }
    
    sum += channel_max;
    
    // Apply final bias and divisor
    output[out_idx] = (sum / divisor) + total_bias;
}

void fused_op_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor conv_bias,
    int N, int C_in, int D, int H, int W,
    int K_D, int K_H, int K_W,
    int S_D, int S_H, int S_W,
    int P_D, int P_H, int P_W,
    int dilation_D, int dilation_H, int dilation_W,
    int D_out_conv, int H_out_conv, int W_out_conv,
    
    int P_D_max, int P_H_max, int P_W_max,
    int K_D_max, int K_H_max, int K_W_max,
    int S_D_max, int S_H_max, int S_W_max,
    int dilation_D_max, int dilation_H_max, int dilation_W_max,
    int D_out_max, int H_out_max, int W_out_max,
    
    int D_out_adapt, int H_out_adapt, int W_out_adapt,
    
    float divisor, float total_bias,
    torch::Tensor output) 
{
    int threads = 256;
    int blocks = (N * D_out_adapt * H_out_adapt * W_out_adapt + threads - 1) / threads;
    
    fused_pipeline_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), conv_bias.data_ptr<float>(),
        N, C_in, D, H, W,
        K_D, K_H, K_W,
        S_D, S_H, S_W,
        P_D, P_H, P_W,
        dilation_D, dilation_H, dilation_W,
        D_out_conv, H_out_conv, W_out_conv,
        
        P_D_max, P_H_max, P_W_max,
        K_D_max, K_H_max, K_W_max,
        S_D_max, S_H_max, S_W_max,
        dilation_D_max, dilation_H_max, dilation_W_max,
        D_out_max, H_out_max, W_out_max,
        
        D_out_adapt, H_out_adapt, W_out_adapt,
        
        divisor, total_bias,
        output.data_ptr<float>()
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor conv_bias,
    int N, int C_in, int D, int H, int W,
    int K_D, int K_H, int K_W,
    int S_D, int S_H, int S_W,
    int P_D, int P_H, int P_W,
    int dilation_D, int dilation_H, int dilation_W,
    int D_out_conv, int H_out_conv, int W_out_conv,
    
    int P_D_max, int P_H_max, int P_W_max,
    int K_D_max, int K_H_max, int K_W_max,
    int S_D_max, int S_H_max, int S_W_max,
    int dilation_D_max, int dilation_H_max, int dilation_W_max,
    int D_out_max, int H_out_max, int W_out_max,
    
    int D_out_adapt, int H_out_adapt, int W_out_adapt,
    
    float divisor, float total_bias,
    torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv3D->MaxPool3D->AdaptiveAvgPool3D->Reduce");
}
"""

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
    
    # Input dimensions
    N, C_in, D, H, W = x.shape
    
    # Conv3D parameters
    K_D, K_H, K_W = conv_weight.shape[-3:]
    S_D, S_H, S_W = conv_stride if isinstance(conv_stride, (tuple, list)) else (conv_stride, conv_stride, conv_stride)
    P_D, P_H, P_W = conv_padding if isinstance(conv_padding, (tuple, list)) else (conv_padding, conv_padding, conv_padding)
    dilation_D, dilation_H, dilation_W = conv_dilation if isinstance(conv_dilation, (tuple, list)) else (conv_dilation, conv_dilation, conv_dilation)
    
    # Compute conv3d output sizes
    D_out_conv = (D + 2 * P_D - dilation_D * (K_D - 1) - 1) // S_D + 1
    H_out_conv = (H + 2 * P_H - dilation_H * (K_H - 1) - 1) // S_H + 1
    W_out_conv = (W + 2 * P_W - dilation_W * (K_W - 1) - 1) // S_W + 1
    
    # MaxPool3D parameters
    K_D_max, K_H_max, K_W_max = max_pool_kernel_size if isinstance(max_pool_kernel_size, (tuple, list)) else (max_pool_kernel_size, max_pool_kernel_size, max_pool_kernel_size)
    S_D_max, S_H_max, S_W_max = max_pool_stride if isinstance(max_pool_stride, (tuple, list)) else (max_pool_stride, max_pool_stride, max_pool_stride)
    P_D_max, P_H_max, P_W_max = max_pool_padding if isinstance(max_pool_padding, (tuple, list)) else (max_pool_padding, max_pool_padding, max_pool_padding)
    dilation_D_max, dilation_H_max, dilation_W_max = max_pool_dilation if isinstance(max_pool_dilation, (tuple, list)) else (max_pool_dilation, max_pool_dilation, max_pool_dilation)
    
    # Compute maxpool3d output sizes
    D_out_max = (D_out_conv + 2 * P_D_max - dilation_D_max * (K_D_max - 1) - 1) // S_D_max + 1
    H_out_max = (H_out_conv + 2 * P_H_max - dilation_H_max * (K_H_max - 1) - 1) // S_H_max + 1
    W_out_max = (W_out_conv + 2 * P_W_max - dilation_W_max * (K_W_max - 1) - 1) // S_W_max + 1
    
    # AdaptiveAvgPool3D parameters
    D_out_adapt, H_out_adapt, W_out_adapt = global_avg_pool_output_size
    
    # Total bias
    total_bias = bias.sum().item()
    
    # Output tensor
    output = torch.empty([N, D_out_adapt, H_out_adapt, W_out_adapt], device=x.device, dtype=x.dtype)
    
    # Launch fused kernel
    fused_ext.fused_op(
        x, conv_weight, conv_bias,
        N, C_in, D, H, W,
        K_D, K_H, K_W,
        S_D, S_H, S_W,
        P_D, P_H, P_W,
        dilation_D, dilation_H, dilation_W,
        D_out_conv, H_out_conv, W_out_conv,
        
        P_D_max, P_H_max, P_W_max,
        K_D_max, K_H_max, K_W_max,
        S_D_max, S_H_max, S_W_max,
        dilation_D_max, dilation_H_max, dilation_W_max,
        D_out_max, H_out_max, W_out_max,
        
        D_out_adapt, H_out_adapt, W_out_adapt,
        
        float(divisor), float(total_bias),
        output
    )
    
    return output
