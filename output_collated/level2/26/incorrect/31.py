# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_043234/code_6.py
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

# ------------------------------------------------------------
# Custom CUDA kernels for ConvTranspose3D and fused operation
# ------------------------------------------------------------

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float hardswish_impl(float x) {
    // hardswish(x) = x * clamp(x + 3, 0, 6) / 6
    float relu6_val = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * relu6_val * 0.16666667f; // 1/6
}

__global__ void fused_add_hardswish_kernel(
    const float* __restrict__ conv_out,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    const int numel) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float x = conv_out[idx] + add_input[idx];
        output[idx] = x * hardswish_impl(x);
    }
}

// ConvTranspose3D kernel implementation
__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int out_depth,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride,
    const int padding) {
    
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (out_idx >= total_elements) return;
    
    const int w = out_idx % out_width;
    const int h = (out_idx / out_width) % out_height;
    const int d = (out_idx / (out_width * out_height)) % out_depth;
    const int c_out = (out_idx / (out_width * out_height * out_depth)) % out_channels;
    const int b = out_idx / (out_width * out_height * out_depth * out_channels);
    
    float sum = 0.0f;
    
    // Calculate input position that could contribute to this output position
    const int in_d_start = max(0, (d + padding - kernel_size + 1 + stride - 1) / stride);
    const int in_d_end = min(in_depth - 1, (d + padding) / stride);
    
    const int in_h_start = max(0, (h + padding - kernel_size + 1 + stride - 1) / stride);
    const int in_h_end = min(in_height - 1, (h + padding) / stride);
    
    const int in_w_start = max(0, (w + padding - kernel_size + 1 + stride - 1) / stride);
    const int in_w_end = min(in_width - 1, (w + padding) / stride);
    
    for (int in_d = in_d_start; in_d <= in_d_end; ++in_d) {
        for (int in_h = in_h_start; in_h <= in_h_end; ++in_h) {
            for (int in_w = in_w_start; in_w <= in_w_end; ++in_w) {
                // Calculate kernel position
                const int kd = d + padding - in_d * stride;
                const int kh = h + padding - in_h * stride;
                const int kw = w + padding - in_w * stride;
                
                if (kd >= 0 && kd < kernel_size && 
                    kh >= 0 && kh < kernel_size && 
                    kw >= 0 && kw < kernel_size) {
                    
                    const int input_idx = b * (in_channels * in_depth * in_height * in_width) +
                                          ((0 * in_depth + in_d) * in_height + in_h) * in_width + in_w;
                    
                    const int weight_idx = c_out * (in_channels * kernel_size * kernel_size * kernel_size) +
                                           ((0 * kernel_size + kd) * kernel_size + kh) * kernel_size + kw;
                    
                    for (int c_in = 0; c_in < in_channels; ++c_in) {
                        const int in_idx = input_idx + c_in * (in_depth * in_height * in_width);
                        const int w_idx = weight_idx + c_in * (kernel_size * kernel_size * kernel_size);
                        sum += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
    }
    
    output[out_idx] = sum + bias[c_out];
}

void launch_fused_add_hardswish(
    const at::Tensor& conv_out,
    const at::Tensor& add_input,
    at::Tensor& output) {
    
    const int numel = conv_out.numel();
    const dim3 block(256);
    const dim3 grid((numel + block.x - 1) / block.x);
    
    fused_add_hardswish_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        conv_out.data_ptr<float>(),
        add_input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "CUDA kernel failed: ", cudaGetErrorString(cudaGetLastError()));
}

void launch_conv_transpose3d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    const int stride,
    const int padding) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);
    
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size;
    const int out_height = (in_height - 1) * stride - 2 * padding + kernel_size;
    const int out_width = (in_width - 1) * stride - 2 * padding + kernel_size;
    
    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    const dim3 block(256);
    const dim3 grid((total_elements + block.x - 1) / block.x);
    
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
"""

# --- C++ Binding ---
cpp_source = r"""
#include <torch/extension.h>

void launch_fused_add_hardswish(
    const at::Tensor& conv_out,
    const at::Tensor& add_input,
    at::Tensor& output);

void launch_conv_transpose3d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    const int stride,
    const int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_add_hardswish", &launch_fused_add_hardswish, "Fused Add + HardSwish");
    m.def("conv_transpose3d", &launch_conv_transpose3d, "ConvTranspose3D");
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
    # Calculate output dimensions for conv transpose
    in_depth, in_height, in_width = x.shape[2], x.shape[3], x.shape[4]
    kernel_size = conv_transpose_weight.shape[2]
    
    out_depth = (in_depth - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    out_height = (in_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    out_width = (in_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    
    # Create output tensor for conv transpose
    conv_output = torch.empty(
        x.shape[0], conv_transpose_weight.shape[0], 
        out_depth, out_height, out_width, 
        device=x.device, dtype=x.dtype
    )
    
    # Launch custom ConvTranspose3D kernel
    fused_ext.conv_transpose3d(
        x, conv_transpose_weight, conv_transpose_bias, conv_output,
        conv_transpose_stride, conv_transpose_padding
    )
    
    # Allocate output tensor for fused operation
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
