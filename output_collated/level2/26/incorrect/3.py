# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_040131/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x', 'add_input']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- Optimized CUDA Kernels with Grid-Stride Loops ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float hardswish_impl(float x) {
    // hardswish(x) = x * clamp(x + 3, 0, 6) / 6
    float relu6_val = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * relu6_val * 0.16666667f; // 1/6
}

// Optimized ConvTranspose3D kernel
__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding) {
    
    // Grid-stride loop for better utilization
    int total_threads = batch_size * out_channels * out_depth * out_height * out_width;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; 
         tid < total_threads; 
         tid += blockDim.x * gridDim.x) {
        
        // Calculate output indices
        int ow = tid % out_width;
        int oh = (tid / out_width) % out_height;
        int od = (tid / (out_width * out_height)) % out_depth;
        int oc = (tid / (out_width * out_height * out_depth)) % out_channels;
        int b = tid / (out_width * out_height * out_depth * out_channels);
        
        if (b >= batch_size) continue;
        
        // Calculate input position
        int id = od - padding;
        int ih = oh - padding;
        int iw = ow - padding;
        
        float sum = 0.0f;
        
        // Loop through kernel
        for (int kd = 0; kd < kernel_size; kd++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int in_d = id - kd * stride;
                    int in_h = ih - kh * stride;
                    int in_w = iw - kw * stride;
                    
                    // Check bounds
                    if (in_d >= 0 && in_d < in_depth &&
                        in_h >= 0 && in_h < in_height &&
                        in_w >= 0 && in_w < in_width) {
                        
                        int input_idx = b * (in_channels * in_depth * in_height * in_width) +
                                       oc * (in_depth * in_height * in_width) +
                                       in_d * (in_height * in_width) +
                                       in_h * in_width +
                                       in_w;
                                       
                        int weight_idx = oc * (in_channels * kernel_size * kernel_size * kernel_size) +
                                        kd * (kernel_size * kernel_size) +
                                        kh * kernel_size +
                                        kw;
                                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Add bias
        output[tid] = sum + bias[oc];
    }
}

__global__ void fused_add_hardswish_kernel(
    const float* __restrict__ conv_out,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    const int numel) {

    // Grid-stride loop for better utilization
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < numel; 
         idx += blockDim.x * gridDim.x) {
        float x = conv_out[idx] + add_input[idx];
        output[idx] = x * hardswish_impl(x);
    }
}

void launch_conv_transpose3d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    int kernel_size,
    int stride,
    int padding) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    int out_channels = output.size(1);
    int out_depth = output.size(2);
    int out_height = output.size(3);
    int out_width = output.size(4);
    
    // Optimize grid configuration
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    const int block_size = 512;
    const int max_blocks = at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 4;
    const int num_blocks = min(max_blocks, (total_elements + block_size - 1) / block_size);
    
    const dim3 block(block_size);
    const dim3 grid(num_blocks);
    
    conv_transpose3d_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_depth,
        in_height,
        in_width,
        out_depth,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "ConvTranspose3D CUDA kernel failed: ", cudaGetErrorString(cudaGetLastError()));
}

void launch_fused_add_hardswish(
    const at::Tensor& conv_out,
    const at::Tensor& add_input,
    at::Tensor& output) {
    
    const int numel = conv_out.numel();
    
    // Optimize grid configuration
    const int block_size = 512;
    const int max_blocks = at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 4;
    const int num_blocks = min(max_blocks, (numel + block_size - 1) / block_size);
    
    const dim3 block(block_size);
    const dim3 grid(num_blocks);
    
    fused_add_hardswish_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        conv_out.data_ptr<float>(),
        add_input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "Fused Add+HardSwish CUDA kernel failed: ", cudaGetErrorString(cudaGetLastError()));
}
"""

# --- C++ Binding ---
cpp_source = r"""
#include <torch/extension.h>

void launch_conv_transpose3d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    int kernel_size,
    int stride,
    int padding);

void launch_fused_add_hardswish(
    const at::Tensor& conv_out,
    const at::Tensor& add_input,
    at::Tensor& output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose3d", &launch_conv_transpose3d, "ConvTranspose3D CUDA kernel");
    m.def("fused_add_hardswish", &launch_fused_add_hardswish, "Fused Add + HardSwish");
}
"""

# --- Compile Extension ---
fused_ext = load_inline(
    name='fused_op_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    add_input,
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
    # Custom ConvTranspose3D using optimized CUDA kernel
    batch_size, in_channels, D, H, W = x.shape
    out_channels = conv_transpose_weight.shape[0]
    kernel_size = conv_transpose_weight.shape[2]
    
    # Calculate output dimensions
    out_D = (D - 1) * conv_transpose_stride + kernel_size - 2 * conv_transpose_padding + conv_transpose_output_padding
    out_H = (H - 1) * conv_transpose_stride + kernel_size - 2 * conv_transpose_padding + conv_transpose_output_padding
    out_W = (W - 1) * conv_transpose_stride + kernel_size - 2 * conv_transpose_padding + conv_transpose_output_padding
    
    # Allocate output tensor
    conv_output = torch.empty(batch_size, out_channels, out_D, out_H, out_W, device=x.device, dtype=x.dtype)
    
    # Launch optimized ConvTranspose3D kernel
    fused_ext.conv_transpose3d(
        x, conv_transpose_weight, conv_transpose_bias, conv_output,
        kernel_size, conv_transpose_stride, conv_transpose_padding
    )
    
    # Allocate final output tensor
    output = torch.empty_like(conv_output)
    
    # Launch fused CUDA kernel for element-wise operations
    fused_ext.fused_add_hardswish(conv_output, add_input, output)
    
    return output

# --- Test Configuration ---
batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]

def get_inputs():
    return [
        torch.rand(batch_size, in_channels, D, H, W, device='cuda'),
        torch.rand(batch_size, out_channels, D*stride, H*stride, W*stride, device='cuda')
    ]

# Move model parameters to GPU for consistency
def move_params_to_cuda(params_dict):
    for k, v in params_dict.items():
        if isinstance(v, torch.Tensor):
            params_dict[k] = v.cuda()

# Example usage (for testing)
if __name__ == "__main__":
    import time
    
    # Create test inputs on GPU
    inputs = get_inputs()
    x_gpu, add_input_gpu = inputs
    
    # Create parameters and move to GPU
    params = {
        "conv_transpose_weight": torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size, device='cuda'),
        "conv_transpose_bias": torch.randn(out_channels, device='cuda'),
        "conv_transpose_stride": stride,
        "conv_transpose_padding": padding,
        "conv_transpose_output_padding": output_padding,
        "conv_transpose_groups": 1,
        "conv_transpose_dilation": 1,
        "bias": torch.randn(bias_shape, device='cuda')
    }
    
    # Warmup run
    with torch.no_grad():
        _ = functional_model(x_gpu, add_input_gpu, **params)
    
    # Benchmark
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    with torch.no_grad():
        for _ in range(10):
            y = functional_model(x_gpu, add_input_gpu, **params)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Average time per iteration: {elapsed_time_ms / 10:.3f} ms")
    print(f"Output shape: {y.shape}")
    print("Fused operation completed successfully.")
