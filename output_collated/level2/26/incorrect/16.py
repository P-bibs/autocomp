# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_041736/code_0.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- CUDA Kernel with Grid-Stride Loop and Custom ConvTranspose3D ---
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

    // Grid-stride loop: each thread processes multiple elements
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < numel; 
         idx += blockDim.x * gridDim.x) {
        float x = conv_out[idx] + add_input[idx];
        output[idx] = x * hardswish_impl(x);
    }
}

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
    
    const int out_channels_per_group = out_channels;
    const int in_channels_per_group = in_channels;
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < batch_size * out_channels * out_depth * out_height * out_width;
         index += blockDim.x * gridDim.x) {
        
        const int w = index % out_width;
        const int h = (index / out_width) % out_height;
        const int d = (index / (out_width * out_height)) % out_depth;
        const int oc = (index / (out_width * out_height * out_depth)) % out_channels;
        const int n = index / (out_width * out_height * out_depth * out_channels);
        
        float sum = 0.0f;
        
        // Calculate input coordinates that would contribute to this output point
        const int start_kd = max(0, (d + padding - kernel_size + 1 + stride - 1) / stride);
        const int end_kd = min(in_depth, (d + padding + stride) / stride);
        const int start_kh = max(0, (h + padding - kernel_size + 1 + stride - 1) / stride);
        const int end_kh = min(in_height, (h + padding + stride) / stride);
        const int start_kw = max(0, (w + padding - kernel_size + 1 + stride - 1) / stride);
        const int end_kw = min(in_width, (w + padding + stride) / stride);
        
        for (int kd = start_kd; kd < end_kd; ++kd) {
            for (int kh = start_kh; kh < end_kh; ++kh) {
                for (int kw = start_kw; kw < end_kw; ++kw) {
                    const int kd_kernel = d + padding - kd * stride;
                    const int kh_kernel = h + padding - kh * stride;
                    const int kw_kernel = w + padding - kw * stride;
                    
                    if (kd_kernel >= 0 && kd_kernel < kernel_size &&
                        kh_kernel >= 0 && kh_kernel < kernel_size &&
                        kw_kernel >= 0 && kw_kernel < kernel_size) {
                        
                        const int weight_idx = oc * (in_channels * kernel_size * kernel_size * kernel_size) +
                                              0 * (kernel_size * kernel_size * kernel_size) +  // Assuming group=1
                                              kd_kernel * (kernel_size * kernel_size) +
                                              kh_kernel * kernel_size +
                                              kw_kernel;
                        
                        const int input_idx = n * (in_channels * in_depth * in_height * in_width) +
                                             0 * (in_depth * in_height * in_width) +  // Assuming group=1
                                             kd * (in_height * in_width) +
                                             kh * in_width +
                                             kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        const int bias_idx = oc;
        output[index] = sum + bias[bias_idx];
    }
}

void launch_fused_add_hardswish(
    const at::Tensor& conv_out,
    const at::Tensor& add_input,
    at::Tensor& output) {
    
    const int numel = conv_out.numel();
    const dim3 block(256);
    // Reduce grid size - with grid-stride loops we need fewer blocks
    const dim3 grid(min((numel + block.x - 1) / block.x, 65535U)); // Cap at max grid size
    
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
    
    const int out_depth = output.size(2);
    const int out_height = output.size(3);
    const int out_width = output.size(4);
    
    const int numel = batch_size * out_channels * out_depth * out_height * out_width;
    const dim3 block(256);
    const dim3 grid(min((numel + block.x - 1) / block.x, 65535U));
    
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
    m.def("conv_transpose3d", &launch_conv_transpose3d, "ConvTranspose3D custom implementation");
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
    # Custom ConvTranspose3D implementation
    # Calculate output dimensions
    out_d = (x.shape[2] - 1) * conv_transpose_stride + conv_transpose_dilation * (conv_transpose_weight.shape[2] - 1) + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    out_h = (x.shape[3] - 1) * conv_transpose_stride + conv_transpose_dilation * (conv_transpose_weight.shape[3] - 1) + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    out_w = (x.shape[4] - 1) * conv_transpose_stride + conv_transpose_dilation * (conv_transpose_weight.shape[4] - 1) + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    
    conv_out = torch.empty((x.shape[0], conv_transpose_weight.shape[0], out_d, out_h, out_w), device=x.device, dtype=x.dtype)
    
    # Launch custom ConvTranspose3D kernel
    fused_ext.conv_transpose3d(
        x, conv_transpose_weight, conv_transpose_bias, conv_out,
        conv_transpose_stride, conv_transpose_padding
    )
    
    # Allocate output tensor
    output = torch.empty_like(conv_out)
    
    # Launch fused CUDA kernel for element-wise operations
    fused_ext.fused_add_hardswish(conv_out, add_input, output)
    
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
