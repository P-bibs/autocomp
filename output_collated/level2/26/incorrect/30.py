# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_043234/code_4.py
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

# --- CUDA Kernel for Fused Operation with Grid-Stride & Vectorization ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float hardswish_activation(float x) {
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
         idx += gridDim.x * blockDim.x) {
        
        float x = conv_out[idx] + add_input[idx];
        output[idx] = hardswish_activation(x);
    }
}

__global__ void fused_add_hardswish_vectorized_kernel(
    const float4* __restrict__ conv_out,
    const float4* __restrict__ add_input,
    float4* __restrict__ output,
    const int numel_vec) {

    // Vectorized version using float4 for 4x better bandwidth utilization
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < numel_vec; 
         idx += gridDim.x * blockDim.x) {
        
        float4 conv_v = conv_out[idx];
        float4 add_v = add_input[idx];
        
        float4 result;
        result.x = hardswish_activation(conv_v.x + add_v.x);
        result.y = hardswish_activation(conv_v.y + add_v.y);
        result.z = hardswish_activation(conv_v.z + add_v.z);
        result.w = hardswish_activation(conv_v.w + add_v.w);
        
        output[idx] = result;
    }
}

// Custom conv transpose 3d kernel
__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int D, int H, int W,
    int kD, int kH, int kW,
    int stride, int padding,
    int output_padding) {
    
    int out_D = (D - 1) * stride - 2 * padding + kD + output_padding;
    int out_H = (H - 1) * stride - 2 * padding + kH + output_padding;
    int out_W = (W - 1) * stride - 2 * padding + kW + output_padding;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * out_channels * out_D * out_H * out_W;
    
    for (int idx = tid; idx < total_threads; idx += gridDim.x * blockDim.x) {
        int n = idx / (out_channels * out_D * out_H * out_W);
        int c_out = (idx / (out_D * out_H * out_W)) % out_channels;
        int d_out = (idx / (out_H * out_W)) % out_D;
        int h_out = (idx / out_W) % out_H;
        int w_out = idx % out_W;
        
        float sum = 0.0f;
        
        // Loop through kernel dimensions
        for (int kd = 0; kd < kD; kd++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    // Calculate corresponding input position
                    int d_in = d_out + padding - kd;
                    int h_in = h_out + padding - kh;
                    int w_in = w_out + padding - kw;
                    
                    // Check if we're in valid input range
                    if (d_in >= 0 && d_in < D*stride && d_in % stride == 0 &&
                        h_in >= 0 && h_in < H*stride && h_in % stride == 0 &&
                        w_in >= 0 && w_in < W*stride && w_in % stride == 0) {
                        
                        d_in /= stride;
                        h_in /= stride;
                        w_in /= stride;
                        
                        // Loop through input channels
                        for (int c_in = 0; c_in < in_channels; c_in++) {
                            int input_idx = n * (in_channels * D * H * W) +
                                          c_in * (D * H * W) +
                                          d_in * (H * W) +
                                          h_in * W +
                                          w_in;
                          
                            int weight_idx = c_out * (in_channels * kD * kH * kW) +
                                           c_in * (kD * kH * kW) +
                                           kd * (kH * kW) +
                                           kh * kW +
                                           kw;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        output[idx] = sum + bias[c_out];
    }
}

void launch_fused_add_hardswish(
    const at::Tensor& conv_out,
    const at::Tensor& add_input,
    at::Tensor& output) {
    
    const int numel = conv_out.numel();
    
    // Use maximum occupancy configuration
    const int block_size = 512;
    int grid_size = (numel + block_size - 1) / block_size;
    
    // Cap grid size to avoid excessive overhead on smaller problems
    if (numel > 1000000) {
        grid_size = min(grid_size, 65536);  // Max grid dimension is 65535
    }
    
    const dim3 block(block_size);
    const dim3 grid(grid_size);
    
    // Check if we can use vectorized version (numel must be divisible by 4)
    if (numel % 4 == 0) {
        // Use vectorized kernel for 4x bandwidth improvement
        fused_add_hardswish_vectorized_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const float4*>(conv_out.data_ptr<float>()),
            reinterpret_cast<const float4*>(add_input.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            numel / 4
        );
    } else {
        // Fall back to scalar version for non-aligned sizes
        fused_add_hardswish_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            conv_out.data_ptr<float>(),
            add_input.data_ptr<float>(),
            output.data_ptr<float>(),
            numel
        );
    }
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "CUDA kernel failed: ", cudaGetErrorString(cudaGetLastError()));
}

void launch_conv_transpose3d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    int stride, int padding, int output_padding) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);
    
    int block_size = 512;
    int total_elements = output.numel();
    int grid_size = (total_elements + block_size - 1) / block_size;
    grid_size = min(grid_size, 65536);
    
    const dim3 block(block_size);
    const dim3 grid(grid_size);
    
    conv_transpose3d_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        weight.size(0), // out_channels
        D, H, W,
        kD, kH, kW,
        stride, padding, output_padding
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "Conv transpose 3d kernel failed: ", cudaGetErrorString(cudaGetLastError()));
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
    int stride, int padding, int output_padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_add_hardswish", &launch_fused_add_hardswish, "Fused Add + HardSwish with Grid-Stride");
    m.def("conv_transpose3d", &launch_conv_transpose3d, "Custom Conv Transpose 3D");
}
"""

# --- Compile Extension ---
fused_ext = load_inline(
    name='fused_op_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-Xptxas', '-O3'],
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
    batch_size, in_channels, D, H, W = x.shape
    out_channels = conv_transpose_weight.shape[0]
    
    # Calculate output dimensions
    out_D = (D - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_weight.shape[2] + conv_transpose_output_padding
    out_H = (H - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_weight.shape[3] + conv_transpose_output_padding
    out_W = (W - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_weight.shape[4] + conv_transpose_output_padding
    
    # Allocate output tensor
    conv_output = torch.empty(batch_size, out_channels, out_D, out_H, out_W, device=x.device, dtype=x.dtype)
    
    # Launch custom conv transpose 3d kernel
    fused_ext.conv_transpose3d(
        x, conv_transpose_weight, conv_transpose_bias, 
        conv_output, conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding
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

def move_params_to_cuda(params_dict):
    for k, v in params_dict.items():
        if isinstance(v, torch.Tensor):
            params_dict[k] = v.cuda()

if __name__ == "__main__":
    import time
    
    inputs = get_inputs()
    x_gpu, add_input_gpu = inputs
    
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
    
    with torch.no_grad():
        _ = functional_model(x_gpu, add_input_gpu, **params)
    
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
