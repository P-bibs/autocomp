# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_033345/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'max_pool1_kernel_size', 'max_pool1_stride', 'max_pool1_padding', 'max_pool1_dilation', 'max_pool1_ceil_mode', 'max_pool1_return_indices', 'max_pool2_kernel_size', 'max_pool2_stride', 'max_pool2_padding', 'max_pool2_dilation', 'max_pool2_ceil_mode', 'max_pool2_return_indices']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)

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
    # State for conv_transpose (nn.ConvTranspose3d)
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
    # State for max_pool1 (nn.MaxPool3d)
    state_kwargs['max_pool1_kernel_size'] = model.max_pool1.kernel_size
    state_kwargs['max_pool1_stride'] = model.max_pool1.stride
    state_kwargs['max_pool1_padding'] = model.max_pool1.padding
    state_kwargs['max_pool1_dilation'] = model.max_pool1.dilation
    state_kwargs['max_pool1_ceil_mode'] = model.max_pool1.ceil_mode
    state_kwargs['max_pool1_return_indices'] = model.max_pool1.return_indices
    # State for max_pool2 (nn.MaxPool3d)
    state_kwargs['max_pool2_kernel_size'] = model.max_pool2.kernel_size
    state_kwargs['max_pool2_stride'] = model.max_pool2.stride
    state_kwargs['max_pool2_padding'] = model.max_pool2.padding
    state_kwargs['max_pool2_dilation'] = model.max_pool2.dilation
    state_kwargs['max_pool2_ceil_mode'] = model.max_pool2.ceil_mode
    state_kwargs['max_pool2_return_indices'] = model.max_pool2.return_indices
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

# --- CUDA Kernel Code ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

// Helper to get linear index from 5D coordinates for NCDHW layout
__device__ __forceinline__ int get_index_5d(int n, int c, int d, int h, int w, 
                                            int C, int D, int H, int W) {
    return ((n * C + c) * D + d) * H * W + h * W + w;
}

// Helper to get linear index from 5D coordinates for NDHWC layout (for output)
__device__ __forceinline__ int get_index_5d_out(int n, int c, int d, int h, int w, 
                                                int C, int D, int H, int W) {
    // We only ever write to c=0 in the output, so C=1 for output tensor
    return (n * 1 + c) * D * H * W + (d * H + h) * W + w;
}

__global__ void fused_conv_transpose3d_sum_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth, int input_height, int input_width,
    int kernel_size_d, int kernel_size_h, int kernel_size_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int output_depth, int output_height, int output_width
) {
    // Each thread block handles one output spatial location for all batches
    extern __shared__ float shared_mem[];

    int tid = threadIdx.x;
    int spatial_id = blockIdx.x;
    int total_spatial = output_depth * output_height * output_width;
    
    if (spatial_id >= total_spatial) return;

    int out_d = (spatial_id / (output_height * output_width)) % output_depth;
    int out_h = (spatial_id / output_width) % output_height;
    int out_w = spatial_id % output_width;

    // Shared memory to accumulate partial sums across channels
    float* shared_sum = shared_mem; // Size: batch_size
    
    // Initialize shared memory
    for (int i = tid; i < batch_size; i += blockDim.x) {
        shared_sum[i] = 0.0f;
    }
    __syncthreads();

    // Convolve and sum across output channels
    for (int out_c = 0; out_c < out_channels; out_c++) {
        float channel_sum = 0.0f;
        
        // Iterate through input channels and kernel
        for (int in_c = 0; in_c < in_channels; in_c++) {
            for (int kd = 0; kd < kernel_size_d; kd++) {
                for (int kh = 0; kh < kernel_size_h; kh++) {
                    for (int kw = 0; kw < kernel_size_w; kw++) {
                        // Calculate corresponding input position
                        int in_d = out_d + padding_d - kd * dilation_d;
                        int in_h = out_h + padding_h - kh * dilation_h;
                        int in_w = out_w + padding_w - kw * dilation_w;
                        
                        // Check if it's a valid input position
                        if (in_d % stride_d == 0 && in_h % stride_h == 0 && in_w % stride_w == 0) {
                            in_d /= stride_d;
                            in_h /= stride_h;
                            in_w /= stride_w;
                            
                            if (in_d >= 0 && in_d < input_depth &&
                                in_h >= 0 && in_h < input_height &&
                                in_w >= 0 && in_w < input_width) {
                                
                                // Load input and weight values
                                float input_val = 0.0f;
                                float weight_val = 0.0f;
                                
                                for (int b = tid; b < batch_size; b += blockDim.x) {
                                    int input_idx = get_index_5d(b, in_c, in_d, in_h, in_w,
                                                                in_channels, input_depth, input_height, input_width);
                                    input_val = input[input_idx];
                                    
                                    int weight_idx = get_index_5d(out_c, in_c, kd, kh, kw,
                                                                in_channels, kernel_size_d, kernel_size_h, kernel_size_w);
                                    weight_val = weight[weight_idx];
                                    
                                    // Accumulate partial sum for this channel
                                    channel_sum += input_val * weight_val;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Add bias if this is the first channel
        if (out_c == 0) {
            for (int b = tid; b < batch_size; b += blockDim.x) {
                channel_sum += bias[out_c];
            }
        }
        
        // Accumulate channel sum into shared memory
        for (int b = tid; b < batch_size; b += blockDim.x) {
            shared_sum[b] += channel_sum;
        }
        __syncthreads();
    }

    // Write final sum to output
    for (int b = tid; b < batch_size; b += blockDim.x) {
        int output_idx = get_index_5d_out(b, 0, out_d, out_h, out_w,
                                         1, output_depth, output_height, output_width);
        output[output_idx] = shared_sum[b];
    }
}

void fused_conv_transpose3d_sum_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    // Set the GPU device to match the input tensor
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);

    int out_channels = weight.size(0);
    int kernel_size_d = weight.size(2);
    int kernel_size_h = weight.size(3);
    int kernel_size_w = weight.size(4);

    // Compute output dimensions
    int output_depth = (input_depth - 1) * stride_d - 2 * padding_d + dilation_d * (kernel_size_d - 1) + 1 + output_padding_d;
    int output_height = (input_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_size_h - 1) + 1 + output_padding_h;
    int output_width = (input_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_size_w - 1) + 1 + output_padding_w;

    // Prepare grid and block dimensions
    int total_spatial = output_depth * output_height * output_width;
    dim3 grid(total_spatial);
    dim3 block(min(256, batch_size)); // Adjust block size based on batch size
    
    // Allocate shared memory: one float per batch item
    size_t shared_mem_size = batch_size * sizeof(float);

    // Launch kernel
    fused_conv_transpose3d_sum_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth, input_height, input_width,
        kernel_size_d, kernel_size_h, kernel_size_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        dilation_d, dilation_h, dilation_w,
        output_depth, output_height, output_width
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR(cudaGetErrorString(err));
    }
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_sum_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_sum", &fused_conv_transpose3d_sum_forward, "Fused 3D ConvTranspose and Sum");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose3d_sum_ext',
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
    max_pool1_kernel_size,
    max_pool1_stride,
    max_pool1_padding,
    max_pool1_dilation,
    max_pool1_ceil_mode,
    max_pool1_return_indices,
    max_pool2_kernel_size,
    max_pool2_stride,
    max_pool2_padding,
    max_pool2_dilation,
    max_pool2_ceil_mode,
    max_pool2_return_indices,
):
    # --- Step 1: Fused ConvTranspose3D + Sum ---
    # Compute output dimensions for conv transpose
    stride_d, stride_h, stride_w = conv_transpose_stride
    padding_d, padding_h, padding_w = conv_transpose_padding
    output_padding_d, output_padding_h, output_padding_w = conv_transpose_output_padding
    dilation_d, dilation_h, dilation_w = conv_transpose_dilation
    
    kernel_size_d = conv_transpose_weight.size(2)
    kernel_size_h = conv_transpose_weight.size(3)
    kernel_size_w = conv_transpose_weight.size(4)
    
    input_depth = x.size(2)
    input_height = x.size(3)
    input_width = x.size(4)
    
    output_depth = (input_depth - 1) * stride_d - 2 * padding_d + dilation_d * (kernel_size_d - 1) + 1 + output_padding_d
    output_height = (input_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_size_h - 1) + 1 + output_padding_h
    output_width = (input_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_size_w - 1) + 1 + output_padding_w
    
    # Create output tensor with shape [batch, 1, depth, height, width]
    output = torch.zeros(x.size(0), 1, output_depth, output_height, output_width, device=x.device, dtype=x.dtype)
    
    # Call our custom fused kernel
    fused_ext.fused_conv_transpose3d_sum(
        x, conv_transpose_weight, conv_transpose_bias, output,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        dilation_d, dilation_h, dilation_w
    )
    
    x = output

    # --- Step 2: Max Pooling Operations ---
    x = F.max_pool3d(x, kernel_size=max_pool1_kernel_size, stride=max_pool1_stride, padding=max_pool1_padding, 
                     dilation=max_pool1_dilation, ceil_mode=max_pool1_ceil_mode, return_indices=max_pool1_return_indices)
    x = F.max_pool3d(x, kernel_size=max_pool2_kernel_size, stride=max_pool2_stride, padding=max_pool2_padding, 
                     dilation=max_pool2_dilation, ceil_mode=max_pool2_ceil_mode, return_indices=max_pool2_return_indices)
    
    return x

# --- Constants (unchanged) ---
batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 32, 32, 32
kernel_size = 5
stride = 2
padding = 2

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
