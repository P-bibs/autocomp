# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_101623/code_0.py
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void fused_conv_min_softmax_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int D, int H, int W,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups,
    int dim,
    int out_D, int out_H, int out_W) {
    
    // Calculate global thread index for output elements
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total number of elements in the final output (after min reduction)
    int reduced_D = (dim == 2) ? out_H * out_W : out_D;
    int reduced_H = (dim == 3) ? out_D * out_W : out_H;
    int reduced_W = (dim == 4) ? out_D * out_H : out_W;
    int total_reduced_elements = batch_size * out_channels * reduced_D * reduced_H * reduced_W;
    
    if (tid >= total_reduced_elements) return;
    
    // Decode position in the reduced output tensor
    int temp = tid;
    int w_out_reduced = temp % reduced_W; temp /= reduced_W;
    int h_out_reduced = temp % reduced_H; temp /= reduced_H;
    int d_out_reduced = temp % reduced_D; temp /= reduced_D;
    int c_out = temp % out_channels; temp /= out_channels;
    int b = temp;
    
    // Map back to original 3D coordinates based on which dimension was reduced
    int d_out, h_out, w_out;
    if (dim == 2) { // Depth reduced
        d_out = 0; // Not used since we're minimizing over this dimension
        h_out = h_out_reduced;
        w_out = w_out_reduced;
    } else if (dim == 3) { // Height reduced
        d_out = d_out_reduced;
        h_out = 0; // Not used
        w_out = w_out_reduced;
    } else if (dim == 4) { // Width reduced
        d_out = d_out_reduced;
        h_out = h_out_reduced;
        w_out = 0; // Not used
    } else {
        return; // Unsupported dimension
    }
    
    // Compute min reduction along the specified dimension
    float min_val = INFINITY;
    
    // Determine the dimension we're reducing over
    int reduce_size;
    if (dim == 2) reduce_size = out_D;
    else if (dim == 3) reduce_size = out_H;
    else if (dim == 4) reduce_size = out_W;
    else return;
    
    // Loop through the reduction dimension
    for (int reduce_idx = 0; reduce_idx < reduce_size; ++reduce_idx) {
        // Calculate actual 3D output position
        int actual_d_out = (dim == 2) ? reduce_idx : d_out;
        int actual_h_out = (dim == 3) ? reduce_idx : h_out;
        int actual_w_out = (dim == 4) ? reduce_idx : w_out;
        
        // Calculate corresponding input position
        int d_in = actual_d_out * stride - padding;
        int h_in = actual_h_out * stride - padding;
        int w_in = actual_w_out * stride - padding;
        
        // Perform convolution for this position
        float conv_result = 0.0f;
        
        // Assuming groups=1 and kernel_size=3 for optimization
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            for (int kd = 0; kd < kernel_size; ++kd) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int in_d = d_in + kd * dilation;
                        int in_h = h_in + kh * dilation;
                        int in_w = w_in + kw * dilation;
                        
                        if (in_d >= 0 && in_d < D && 
                            in_h >= 0 && in_h < H && 
                            in_w >= 0 && in_w < W) {
                            int input_idx = b * (in_channels * D * H * W) + 
                                           c_in * (D * H * W) + 
                                           in_d * (H * W) + 
                                           in_h * W + in_w;
                                           
                            int weight_idx = c_out * (in_channels * kernel_size * kernel_size * kernel_size) +
                                            c_in * (kernel_size * kernel_size * kernel_size) +
                                            kd * (kernel_size * kernel_size) +
                                            kh * kernel_size + kw;
                            
                            conv_result += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        conv_result += bias[c_out];
        min_val = fminf(min_val, conv_result);
    }
    
    // Now compute softmax across channels for this spatial location
    // We use shared memory to collect values from all channels
    extern __shared__ float shared_mem[];
    float* shared_vals = shared_mem;
    float* shared_exp_vals = &shared_mem[out_channels];
    
    // All threads in the block load their channel's value
    shared_vals[c_out] = min_val;
    __syncthreads();
    
    // Only one thread per spatial location computes softmax
    if (c_out == 0) {
        // Find maximum for numerical stability
        float max_val = -INFINITY;
        for (int cc = 0; cc < out_channels; ++cc) {
            max_val = fmaxf(max_val, shared_vals[cc]);
        }
        
        // Compute exponentials and sum
        float sum = 0.0f;
        for (int cc = 0; cc < out_channels; ++cc) {
            shared_exp_vals[cc] = expf(shared_vals[cc] - max_val);
            sum += shared_exp_vals[cc];
        }
        
        // Write normalized values to output
        for (int cc = 0; cc < out_channels; ++cc) {
            int out_idx;
            if (dim == 2) {
                out_idx = b * (out_channels * out_H * out_W) +
                         cc * (out_H * out_W) +
                         h_out_reduced * out_W + w_out_reduced;
            } else if (dim == 3) {
                out_idx = b * (out_channels * out_D * out_W) +
                         cc * (out_D * out_W) +
                         d_out_reduced * out_W + w_out_reduced;
            } else if (dim == 4) {
                out_idx = b * (out_channels * out_D * out_H) +
                         cc * (out_D * out_H) +
                         d_out_reduced * out_H + h_out_reduced;
            }
            output[out_idx] = shared_exp_vals[cc] / sum;
        }
    }
}

void fused_op_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int D, int H, int W,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups,
    int dim,
    int out_D, int out_H, int out_W) {
    
    // Calculate output dimensions after reduction
    int reduced_D = (dim == 2) ? out_H * out_W : out_D;
    int reduced_H = (dim == 3) ? out_D * out_W : out_H;
    int reduced_W = (dim == 4) ? out_D * out_H : out_W;
    int total_reduced_elements = batch_size * out_channels * reduced_D * reduced_H * reduced_W;
    
    int threads_per_block = 256;
    int blocks = (total_reduced_elements + threads_per_block - 1) / threads_per_block;
    
    // Shared memory for softmax: 2 arrays of out_channels floats each
    size_t shared_mem_size = 2 * out_channels * sizeof(float);
    
    fused_conv_min_softmax_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        input, weight, bias, output,
        batch_size, in_channels, out_channels,
        D, H, W, kernel_size, stride, padding, dilation, groups,
        dim, out_D, out_H, out_W
    );
    
    cudaDeviceSynchronize();
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int D, int H, int W,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups,
    int dim,
    int out_D, int out_H, int out_W);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv-Min-Softmax operation");
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
    # Calculate output dimensions
    D, H, W = x.shape[-3:]
    kernel_size = conv_weight.shape[-1]
    out_D = (D + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_H = (H + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_W = (W + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    # Create output tensor with reduced dimension (after min operation along dim)
    if dim == 2:  # Depth dimension
        output_shape = (x.shape[0], conv_weight.shape[0], out_H, out_W)
    elif dim == 3:  # Height dimension
        output_shape = (x.shape[0], conv_weight.shape[0], out_D, out_W)
    elif dim == 4:  # Width dimension
        output_shape = (x.shape[0], conv_weight.shape[0], out_D, out_H)
    else:
        raise ValueError("Unsupported dimension for min reduction")
    
    output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    
    # Call fused operation
    fused_ext.fused_op(
        x.data_ptr(),
        conv_weight.data_ptr(),
        conv_bias.data_ptr(),
        output.data_ptr(),
        x.shape[0],  # batch_size
        x.shape[1],  # in_channels
        conv_weight.shape[0],  # out_channels
        D, H, W,
        kernel_size,
        conv_stride,
        conv_padding,
        conv_dilation,
        conv_groups,
        dim,
        out_D, out_H, out_W
    )
    
    return output

batch_size = 128
in_channels = 3
out_channels = 24  # Increased output channels
D, H, W = 24, 32, 32  # Increased depth
kernel_size = 3
dim = 2  # Dimension along which to apply minimum operation (e.g., depth)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]
