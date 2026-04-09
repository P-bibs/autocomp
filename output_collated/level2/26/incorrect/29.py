# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_043234/code_0.py
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

# --- Optimized CUDA Kernel for Fused ConvTranspose3D + Add + HardSwish ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float hardswish_impl(float x) {
    // hardswish(x) = x * clamp(x + 3, 0, 6) / 6
    float relu6_val = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * relu6_val * 0.16666667f; // 1/6
}

__global__ void fused_conv_transpose3d_add_hardswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_d, const int input_h, const int input_w,
    const int output_d, const int output_h, const int output_w,
    const int kernel_size,
    const int stride,
    const int padding) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    
    if (idx >= total_elements) return;
    
    // Calculate output indices
    const int w_out = idx % output_w;
    const int h_out = (idx / output_w) % output_h;
    const int d_out = (idx / (output_w * output_h)) % output_d;
    const int c_out = (idx / (output_w * output_h * output_d)) % out_channels;
    const int batch = idx / (output_w * output_h * output_d * out_channels);
    
    const int kernel_radius = kernel_size / 2;
    
    float conv_sum = (conv_bias != nullptr) ? conv_bias[c_out] : 0.0f;
    
    // Perform convolution transpose operation
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                // Calculate the corresponding input position
                const int in_d = d_out - (kd - kernel_radius) * stride + 2 * padding;
                const int in_h = h_out - (kh - kernel_radius) * stride + 2 * padding;
                const int in_w = w_out - (kw - kernel_radius) * stride + 2 * padding;
                
                // Check if the input position is valid
                if (in_d >= 0 && in_d < input_d * stride && 
                    in_h >= 0 && in_h < input_h * stride && 
                    in_w >= 0 && in_w < input_w * stride) {
                    
                    // For transposed convolution, we need to map back to input coordinates
                    const int src_d = in_d / stride;
                    const int src_h = in_h / stride;
                    const int src_w = in_w / stride;
                    
                    // Only process if the calculated input position is valid
                    if (src_d >= 0 && src_d < input_d &&
                        src_h >= 0 && src_h < input_h &&
                        src_w >= 0 && src_w < input_w) {
                        
                        for (int c_in = 0; c_in < in_channels; ++c_in) {
                            const int input_idx = ((batch * in_channels + c_in) * input_d + src_d) * input_h * input_w + src_h * input_w + src_w;
                            const int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw;
                            
                            conv_sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Perform fused add + hardswish
    float x = conv_sum + add_input[idx];
    output[idx] = x * hardswish_impl(x);
}

void launch_fused_conv_transpose3d_add_hardswish(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& conv_bias,
    const at::Tensor& add_input,
    at::Tensor& output,
    const int kernel_size,
    const int stride,
    const int padding) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_d = input.size(2);
    const int input_h = input.size(3);
    const int input_w = input.size(4);
    
    const int out_channels = weight.size(0);
    const int output_d = output.size(2);
    const int output_h = output.size(3);
    const int output_w = output.size(4);
    
    const int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    const dim3 block(256);
    const dim3 grid((total_elements + block.x - 1) / block.x);
    
    fused_conv_transpose3d_add_hardswish_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.defined() ? conv_bias.data_ptr<float>() : nullptr,
        add_input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_size,
        stride,
        padding
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "CUDA kernel failed: ", cudaGetErrorString(cudaGetLastError()));
}
"""

# --- C++ Binding ---
cpp_source = r"""
#include <torch/extension.h>

void launch_fused_conv_transpose3d_add_hardswish(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& conv_bias,
    const at::Tensor& add_input,
    at::Tensor& output,
    const int kernel_size,
    const int stride,
    const int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_add_hardswish", &launch_fused_conv_transpose3d_add_hardswish, 
          "Fused ConvTranspose3D + Add + HardSwish");
}
"""

# --- Compile Extension ---
fused_ext = load_inline(
    name='fused_conv_op_ext',
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
    # Direct fused implementation - no intermediate tensors
    batch_size, in_channels, D, H, W = x.shape
    out_channels = conv_transpose_weight.shape[0]
    kernel_size = conv_transpose_weight.shape[2]
    
    # Calculate output dimensions
    output_D = (D - 1) * conv_transpose_stride + kernel_size - 2 * conv_transpose_padding + conv_transpose_output_padding
    output_H = (H - 1) * conv_transpose_stride + kernel_size - 2 * conv_transpose_padding + conv_transpose_output_padding
    output_W = (W - 1) * conv_transpose_stride + kernel_size - 2 * conv_transpose_padding + conv_transpose_output_padding
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, output_D, output_H, output_W, device=x.device, dtype=x.dtype)
    
    # Launch fused CUDA kernel
    fused_ext.fused_conv_transpose3d_add_hardswish(
        x, conv_transpose_weight, conv_transpose_bias, add_input, output,
        kernel_size, conv_transpose_stride, conv_transpose_padding
    )
    
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
