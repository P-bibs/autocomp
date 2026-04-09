# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_040131/code_4.py
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

# --- Optimized CUDA Kernels ---
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

// HardSwish activation implementation
__device__ __forceinline__ float hardswish_impl(float x) {
    float relu6_val = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * relu6_val * 0.16666667f; // 1/6
}

// Vectorized fused add + hardswish kernel using float4
__global__ void vectorized_fused_add_hardswish_kernel(
    const float4* __restrict__ conv_out,
    const float4* __restrict__ add_input,
    float4* __restrict__ output,
    const int numel_vec4) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel_vec4) {
        float4 x_vec = conv_out[idx];
        float4 y_vec = add_input[idx];
        
        // Perform element-wise addition and hardswish
        float4 result;
        result.x = (x_vec.x + y_vec.x) * hardswish_impl(x_vec.x + y_vec.x);
        result.y = (x_vec.y + y_vec.y) * hardswish_impl(x_vec.y + y_vec.y);
        result.z = (x_vec.z + y_vec.z) * hardswish_impl(x_vec.z + y_vec.z);
        result.w = (x_vec.w + y_vec.w) * hardswish_impl(x_vec.w + y_vec.w);
        
        output[idx] = result;
    }
}

// ConvTranspose3D Kernel Implementation (Simplified)
// This is a placeholder for a proper implementation which would be much more complex
// For the scope of this optimization, we'll keep the PyTorch implementation but note where custom kernel would go
__global__ void dummy_conv_transpose3d_kernel(
    const float4* __restrict__ input,
    const float4* __restrict__ weight,
    float4* __restrict__ output,
    const int numel) {
    // Placeholder - in full implementation, this would perform actual transposed convolution
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx]; // Just copy for now to satisfy interface
    }
}

// Launch function for vectorized kernel
void launch_vectorized_fused_add_hardswish(
    const at::Tensor& conv_out,
    const at::Tensor& add_input,
    at::Tensor& output) {
    
    const int numel = conv_out.numel();
    const int numel_vec4 = numel / 4;
    
    TORCH_CHECK(numel % 4 == 0, "Tensor size must be divisible by 4 for float4 operations");
    
    const dim3 block(256);
    const dim3 grid((numel_vec4 + block.x - 1) / block.x);
    
    vectorized_fused_add_hardswish_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const float4*>(conv_out.data_ptr<float>()),
        reinterpret_cast<const float4*>(add_input.data_ptr<float>()),
        reinterpret_cast<float4*>(output.data_ptr<float>()),
        numel_vec4
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "Vectorized fused kernel failed: ", cudaGetErrorString(cudaGetLastError()));
}

// Placeholder for ConvTranspose3D launch
void launch_conv_transpose3d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups,
    std::vector<int64_t> dilation) {
    // In a complete implementation, this would call a custom CUDA kernel
    // For now, we just copy the input to output to satisfy the interface
    output.copy_(input);
}
"""

# --- C++ Bindings ---
cpp_source = r"""
#include <torch/extension.h>
#include <vector>

void launch_vectorized_fused_add_hardswish(
    const at::Tensor& conv_out,
    const at::Tensor& add_input,
    at::Tensor& output);

void launch_conv_transpose3d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups,
    std::vector<int64_t> dilation);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vectorized_fused_add_hardswish", &launch_vectorized_fused_add_hardswish,
          "Vectorized Fused Add + HardSwish");
    m.def("conv_transpose3d", &launch_conv_transpose3d,
          "Custom ConvTranspose3D (placeholder)");
}
"""

# --- Compile Extension ---
optimized_ext = load_inline(
    name='optimized_fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
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
    # Ensure tensors are contiguous and on GPU
    x = x.contiguous()
    add_input = add_input.contiguous()
    
    # Ensure tensor sizes are compatible with float4 vectorization
    total_elements = x.numel()
    if total_elements % 4 != 0:
        # Pad tensors to make them compatible with float4
        padding_needed = 4 - (total_elements % 4)
        x = F.pad(x.view(-1), (0, padding_needed)).view_as(x)
        add_input = F.pad(add_input.view(-1), (0, padding_needed)).view_as(add_input)
    
    # ConvTranspose3D - Using PyTorch's implementation for now
    # In a full optimization, this would be replaced with a custom kernel
    x = F.conv_transpose3d(
        x, conv_transpose_weight, conv_transpose_bias,
        stride=conv_transpose_stride, padding=conv_transpose_padding,
        output_padding=conv_transpose_output_padding, groups=conv_transpose_groups,
        dilation=conv_transpose_dilation
    )
    
    # Ensure output tensor is compatible with float4
    total_elements = x.numel()
    if total_elements % 4 != 0:
        padding_needed = 4 - (total_elements % 4)
        x = F.pad(x.view(-1), (0, padding_needed)).view(*x.shape[:-1], -1)
        add_input = F.pad(add_input.view(-1), (0, padding_needed)).view(*add_input.shape[:-1], -1)
    
    # Allocate output tensor with same size as input
    output = torch.empty_like(x)
    
    # Launch optimized fused kernel
    optimized_ext.vectorized_fused_add_hardswish(x, add_input, output)
    
    # If we padded the tensors, remove the padding from the output
    if output.numel() > total_elements:
        output = output.view(-1)[:-padding_needed].view(*output.shape[:-1], -1)
    
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
    print("Optimized fused operation completed successfully.")
