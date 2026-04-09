# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_040131/code_2.py
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

# --- CUDA Kernel for Fused ConvTranspose3D + Add + HardSwish with Grid-Stride Loop ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float hardswish_impl(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) * 0.16666667f;
}

// Optimized kernel using grid-stride loop
__global__ void fused_convtranspose3d_add_hardswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_D, const int input_H, const int input_W,
    const int output_D, const int output_H, const int output_W,
    const int kernel_size,
    const int stride,
    const int padding) {
    
    const int total_elements = batch_size * out_channels * output_D * output_H * output_W;
    
    // Grid-stride loop: each thread processes multiple elements
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_elements; 
         idx += blockDim.x * gridDim.x) {
        
        const int w_out = idx % output_W;
        const int h_out = (idx / output_W) % output_H;
        const int d_out = (idx / (output_W * output_H)) % output_D;
        const int c_out = (idx / (output_W * output_H * output_D)) % out_channels;
        const int n = idx / (output_W * output_H * output_D * out_channels);

        float conv_result = 0.0f;
        
        // ConvTranspose3D computation
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int d_in = d_out - kd + padding;
                    const int h_in = h_out - kh + padding;
                    const int w_in = w_out - kw + padding;

                    if (d_in >= 0 && d_in < input_D * stride && d_in % stride == 0 &&
                        h_in >= 0 && h_in < input_H * stride && h_in % stride == 0 &&
                        w_in >= 0 && w_in < input_W * stride && w_in % stride == 0) {
                        
                        const int d_in_idx = d_in / stride;
                        const int h_in_idx = h_in / stride;
                        const int w_in_idx = w_in / stride;
                        
                        for (int c_in = 0; c_in < in_channels; ++c_in) {
                            const int input_idx = ((((n * in_channels) + c_in) * input_D + d_in_idx) * input_H + h_in_idx) * input_W + w_in_idx;
                            const int weight_idx = ((((c_out * in_channels) + c_in) * kernel_size + kd) * kernel_size + kh) * kernel_size + kw;
                            conv_result += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }

        // Add bias
        conv_result += bias[c_out];
        
        // Add skip connection and apply HardSwish
        const int output_idx = idx;
        float x = conv_result + add_input[output_idx];
        output[output_idx] = x * hardswish_impl(x);
    }
}

void launch_fused_op(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& add_input,
    at::Tensor& output,
    const int kernel_size,
    const int stride,
    const int padding) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_D = input.size(2);
    const int input_H = input.size(3);
    const int input_W = input.size(4);
    
    const int out_channels = weight.size(0);
    const int output_D = output.size(2);
    const int output_H = output.size(3);
    const int output_W = output.size(4);
    
    const int total_elements = batch_size * out_channels * output_D * output_H * output_W;
    const int threads = 256;
    const int blocks = std::min((total_elements + threads - 1) / threads, 65535); // Cap at max grid size
    
    fused_convtranspose3d_add_hardswish_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        add_input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_D, input_H, input_W,
        output_D, output_H, output_W,
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

void launch_fused_op(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& add_input,
    at::Tensor& output,
    const int kernel_size,
    const int stride,
    const int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_fused_op, "Fused ConvTranspose3D + Add + HardSwish");
}
"""

# --- Compile Extension ---
fused_ext = load_inline(
    name='fused_convtranspose3d_ext',
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
    # Ensure all inputs are on CUDA
    if not x.is_cuda:
        x = x.cuda()
    if not add_input.is_cuda:
        add_input = add_input.cuda()
    if not conv_transpose_weight.is_cuda:
        conv_transpose_weight = conv_transpose_weight.cuda()
    if not conv_transpose_bias.is_cuda:
        conv_transpose_bias = conv_transpose_bias.cuda()
    if not bias.is_cuda:
        bias = bias.cuda()
    
    # Validate group and dilation (only support default values for this custom kernel)
    if conv_transpose_groups != 1 or conv_transpose_dilation != 1:
        raise ValueError("Custom kernel only supports groups=1 and dilation=1")
    
    # Allocate output tensor with correct shape
    batch_size, in_channels, input_D, input_H, input_W = x.shape
    out_channels = conv_transpose_weight.shape[0]
    kernel_size = conv_transpose_weight.shape[2]
    
    output_D = (input_D - 1) * conv_transpose_stride + kernel_size - 2 * conv_transpose_padding + conv_transpose_output_padding
    output_H = (input_H - 1) * conv_transpose_stride + kernel_size - 2 * conv_transpose_padding + conv_transpose_output_padding
    output_W = (input_W - 1) * conv_transpose_stride + kernel_size - 2 * conv_transpose_padding + conv_transpose_output_padding
    
    output = torch.empty((batch_size, out_channels, output_D, output_H, output_W), device=x.device, dtype=x.dtype)
    
    # Launch fused CUDA kernel
    fused_ext.fused_op(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        add_input,
        output,
        kernel_size,
        conv_transpose_stride,
        conv_transpose_padding
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
