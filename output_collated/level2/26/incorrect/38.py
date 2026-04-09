# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_044115/code_2.py
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
from torch.utils.cpp_extension import load_inline

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// Vectorized HardSwish function
__device__ __forceinline__ float4 hardswish_vec(float4 v) {
    auto hardswish_impl = [](float x) {
        float relu6_val = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
        return x * relu6_val * 0.16666667f; // x * relu6(x+3) / 6
    };
    return make_float4(
        hardswish_impl(v.x),
        hardswish_impl(v.y),
        hardswish_impl(v.z),
        hardswish_impl(v.w)
    );
}

__global__ void fused_add_hardswish_vectorized_kernel(
    const float* __restrict__ conv_in,
    const float* __restrict__ add_in,
    float* __restrict__ out,
    const int numel) {

    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    // Vectorized path for aligned data
    if (idx + 3 < numel) {
        float4 a = reinterpret_cast<const float4*>(conv_in)[idx / 4];
        float4 b = reinterpret_cast<const float4*>(add_in)[idx / 4];
        
        float4 sum = make_float4(
            a.x + b.x,
            a.y + b.y,
            a.z + b.z,
            a.w + b.w
        );
        
        reinterpret_cast<float4*>(out)[idx / 4] = hardswish_vec(sum);
    } else {
        // Scalar path for remainder elements
        for (int i = idx; i < numel; ++i) {
            float x = conv_in[i] + add_in[i];
            float relu6_val = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
            out[i] = x * relu6_val * 0.16666667f;
        }
    }
}

void launch_fused_add_hardswish(const at::Tensor& conv_out, const at::Tensor& add_input, at::Tensor& output) {
    const int numel = conv_out.numel();
    const dim3 block(256);
    const dim3 grid((numel + 4 * block.x - 1) / (4 * block.x)); // Each thread handles 4 elements
    
    fused_add_hardswish_vectorized_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        conv_out.data_ptr<float>(),
        add_input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "CUDA kernel failed: ", cudaGetErrorString(cudaGetLastError()));
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_fused_add_hardswish(const at::Tensor& conv_out, const at::Tensor& add_input, at::Tensor& output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_add_hardswish", &launch_fused_add_hardswish, "Vectorized Fused Add + HardSwish");
}
"""

fused_ext = load_inline(
    name='fused_op_vec',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# Custom ConvTranspose3D kernel
conv_transpose3d_cuda = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// CUDA kernel for 3D transposed convolution
template <int KERNEL_SIZE>
__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w) {
    
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * D_out * H_out * W_out;
    
    if (out_idx >= total_elements) return;
    
    int w_out = out_idx % W_out; out_idx /= W_out;
    int h_out = out_idx % H_out; out_idx /= H_out;
    int d_out = out_idx % D_out; out_idx /= D_out;
    int c_out = out_idx % out_channels; out_idx /= out_channels;
    int n = out_idx;
    
    float value = 0.0f;
    
    // Loop over input tensor dimensions and kernel dimensions
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < KERNEL_SIZE; ++kd) {
            for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
                for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                    int d_in = d_out - kd + padding_d;
                    int h_in = h_out - kh + padding_h;
                    int w_in = w_out - kw + padding_w;
                    
                    // Check if the point is in valid input range
                    if (d_in % stride_d == 0 && h_in % stride_h == 0 && w_in % stride_w == 0) {
                        d_in /= stride_d;
                        h_in /= stride_h;
                        w_in /= stride_w;
                        
                        if (d_in >= 0 && d_in < D_in &&
                            h_in >= 0 && h_in < H_in &&
                            w_in >= 0 && w_in < W_in) {
                            
                            int input_idx = ((((n * in_channels + c_in) * D_in + d_in) * H_in + h_in) * W_in + w_in);
                            int weight_idx = (((((c_out * in_channels + c_in) * KERNEL_SIZE + kd) * KERNEL_SIZE + kh) * KERNEL_SIZE + kw));
                            
                            value += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    output[out_idx * out_channels * D_out * H_out * W_out + 
           c_out * D_out * H_out * W_out + 
           d_out * H_out * W_out + 
           h_out * W_out + 
           w_out] = value + bias[c_out];
}

void launch_conv_transpose3d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int D_out = (D_in - 1) * stride_d - 2 * padding_d + kernel_size + output_padding_d;
    int H_out = (H_in - 1) * stride_h - 2 * padding_h + kernel_size + output_padding_h;
    int W_out = (W_in - 1) * stride_w - 2 * padding_w + kernel_size + output_padding_w;
    
    const dim3 block(256);
    const dim3 grid((batch_size * out_channels * D_out * H_out * W_out + block.x - 1) / block.x);
    
    if (kernel_size == 3) {
        conv_transpose3d_kernel<3><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, in_channels, out_channels,
            D_in, H_in, W_in,
            D_out, H_out, W_out,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            output_padding_d, output_padding_h, output_padding_w
        );
    }
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "ConvTranspose3D CUDA kernel failed: ", cudaGetErrorString(cudaGetLastError()));
}
"""

conv_transpose3d_cpp = r"""
#include <torch/extension.h>

void launch_conv_transpose3d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose3d", &launch_conv_transpose3d, "Custom ConvTranspose3D");
}
"""

conv_transpose3d_ext = load_inline(
    name='conv_transpose3d_ext',
    cpp_sources=conv_transpose3d_cpp,
    cuda_sources=conv_transpose3d_cuda,
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
    # Use custom ConvTranspose3D implementation
    stride_d = conv_transpose_stride if isinstance(conv_transpose_stride, int) else conv_transpose_stride[0]
    stride_h = conv_transpose_stride if isinstance(conv_transpose_stride, int) else conv_transpose_stride[1]
    stride_w = conv_transpose_stride if isinstance(conv_transpose_stride, int) else conv_transpose_stride[2]
    
    padding_d = conv_transpose_padding if isinstance(conv_transpose_padding, int) else conv_transpose_padding[0]
    padding_h = conv_transpose_padding if isinstance(conv_transpose_padding, int) else conv_transpose_padding[1]
    padding_w = conv_transpose_padding if isinstance(conv_transpose_padding, int) else conv_transpose_padding[2]
    
    output_padding_d = conv_transpose_output_padding if isinstance(conv_transpose_output_padding, int) else conv_transpose_output_padding[0]
    output_padding_h = conv_transpose_output_padding if isinstance(conv_transpose_output_padding, int) else conv_transpose_output_padding[1]
    output_padding_w = conv_transpose_output_padding if isinstance(conv_transpose_output_padding, int) else conv_transpose_output_padding[2]
    
    # Calculate output dimensions manually to create the output tensor
    batch_size, in_channels, D_in, H_in, W_in = x.shape
    kernel_size = conv_transpose_weight.shape[2]
    D_out = (D_in - 1) * stride_d - 2 * padding_d + kernel_size + output_padding_d
    H_out = (H_in - 1) * stride_h - 2 * padding_h + kernel_size + output_padding_h
    W_out = (W_in - 1) * stride_w - 2 * padding_w + kernel_size + output_padding_w
    out_channels = conv_transpose_weight.shape[0]
    
    # Create output tensor
    conv_out = torch.empty((batch_size, out_channels, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Launch custom conv transpose 3D kernel
    conv_transpose3d_ext.conv_transpose3d(
        x, conv_transpose_weight, conv_transpose_bias, conv_out,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w
    )
    
    # Allocate output tensor for fused operation
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
