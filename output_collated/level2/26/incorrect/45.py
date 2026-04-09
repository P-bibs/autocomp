# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_045050/code_0.py
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

# --- Fused CUDA Kernel for ConvTranspose3D + Add + HardSwish ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float hardswish_impl(float x) {
    // hardswish(x) = x * clamp(x + 3, 0, 6) / 6
    float relu6_val = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * relu6_val * 0.16666667f; // 1/6
}

// Optimized 3D transpose convolution implementation with fused operations
__global__ void fused_convtranspose3d_add_hardswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int output_depth,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding) {
    
    // Calculate output position
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_depth * output_height * output_width) return;
    
    int temp = idx;
    const int w_out = temp % output_width; temp /= output_width;
    const int h_out = temp % output_height; temp /= output_height;
    const int d_out = temp % output_depth; temp /= output_depth;
    const int c_out = temp % out_channels; temp /= out_channels;
    const int batch = temp;
    
    float conv_result = 0.0f;
    if (conv_bias != nullptr) {
        conv_result = conv_bias[c_out];
    }
    
    // For transpose convolution, we need to find which input positions contribute to this output position
    // Work backwards from output to input positions
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // Calculate the input position that would contribute to this output position
                // For transpose convolution: input_pos = (output_pos + padding - kernel_pos) / stride
                int d_in = (d_out + padding - kd);
                int h_in = (h_out + padding - kh);
                int w_in = (w_out + padding - kw);
                
                // Check if the division is exact (no remainder)
                if (d_in >= 0 && h_in >= 0 && w_in >= 0 &&
                    d_in % stride == 0 && h_in % stride == 0 && w_in % stride == 0) {
                    
                    d_in /= stride;
                    h_in /= stride;
                    w_in /= stride;
                    
                    // Check if input position is within bounds
                    if (d_in < input_depth && h_in < input_height && w_in < input_width) {
                        // Accumulate contributions from all input channels
                        for (int c_in = 0; c_in < in_channels; c_in++) {
                            const int input_idx = batch * (in_channels * input_depth * input_height * input_width) +
                                                  c_in * (input_depth * input_height * input_width) +
                                                  d_in * (input_height * input_width) +
                                                  h_in * input_width +
                                                  w_in;
                            
                            const int weight_idx = c_out * (in_channels * kernel_size * kernel_size * kernel_size) +
                                                   c_in * (kernel_size * kernel_size * kernel_size) +
                                                   kd * (kernel_size * kernel_size) +
                                                   kh * kernel_size +
                                                   kw;
                            
                            conv_result += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add operation
    const float add_val = add_input[idx];
    const float x = conv_result + add_val;
    
    // HardSwish operation
    output[idx] = x * hardswish_impl(x);
}

void launch_fused_convtranspose3d_add_hardswish(
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
    const int input_depth = input.size(2);
    const int input_height = input.size(3);
    const int input_width = input.size(4);
    
    const int out_channels = weight.size(0);
    const int output_depth = output.size(2);
    const int output_height = output.size(3);
    const int output_width = output.size(4);
    
    const int numel = batch_size * out_channels * output_depth * output_height * output_width;
    const dim3 block(256);
    const dim3 grid((numel + block.x - 1) / block.x);
    
    fused_convtranspose3d_add_hardswish_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.defined() ? conv_bias.data_ptr<float>() : nullptr,
        add_input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
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

void launch_fused_convtranspose3d_add_hardswish(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& conv_bias,
    const at::Tensor& add_input,
    at::Tensor& output,
    const int kernel_size,
    const int stride,
    const int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_convtranspose3d_add_hardswish", &launch_fused_convtranspose3d_add_hardswish, 
          "Fused ConvTranspose3D + Add + HardSwish");
}
"""

# --- Compile Extension ---
fused_ext = load_inline(
    name='fused_convtranspose_op_ext',
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
    # Calculate output dimensions
    batch_size, in_channels, D, H, W = x.shape
    out_channels = conv_transpose_weight.shape[0]
    kernel_size = conv_transpose_weight.shape[2]
    
    # Calculate output dimensions for conv transpose
    out_D = (D - 1) * conv_transpose_stride + kernel_size - 2 * conv_transpose_padding + conv_transpose_output_padding
    out_H = (H - 1) * conv_transpose_stride + kernel_size - 2 * conv_transpose_padding + conv_transpose_output_padding
    out_W = (W - 1) * conv_transpose_stride + kernel_size - 2 * conv_transpose_padding + conv_transpose_output_padding
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, out_D, out_H, out_W, device=x.device, dtype=x.dtype)
    
    # Launch fused CUDA kernel for all operations
    fused_ext.fused_convtranspose3d_add_hardswish(
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
