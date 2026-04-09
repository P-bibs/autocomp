# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_041736/code_2.py
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

# --- CUDA Kernel for Fused Operation ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float4 hardswish_f4(float4 v) {
    const float inv6 = 0.16666667f;
    float4 res;
    res.x = v.x * fminf(fmaxf(v.x + 3.0f, 0.0f), 6.0f) * inv6;
    res.y = v.y * fminf(fmaxf(v.y + 3.0f, 0.0f), 6.0f) * inv6;
    res.z = v.z * fminf(fmaxf(v.z + 3.0f, 0.0f), 6.0f) * inv6;
    res.w = v.w * fminf(fmaxf(v.w + 3.0f, 0.0f), 6.0f) * inv6;
    return res;
}

__global__ void fused_add_hardswish_vectorized_kernel(
    const float4* __restrict__ conv_out,
    const float4* __restrict__ add_input,
    float4* __restrict__ output,
    const int numel_v4) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel_v4) {
        float4 a = conv_out[idx];
        float4 b = add_input[idx];
        
        float4 res;
        res.x = a.x + b.x;
        res.y = a.y + b.y;
        res.z = a.z + b.z;
        res.w = a.w + b.w;

        output[idx] = hardswish_f4(res);
    }
}

void launch_fused_add_hardswish(
    const at::Tensor& conv_out,
    const at::Tensor& add_input,
    at::Tensor& output) {
    
    const int numel = conv_out.numel();
    const int numel_v4 = numel / 4;
    const dim3 block(256);
    const dim3 grid((numel_v4 + block.x - 1) / block.x);
    
    fused_add_hardswish_vectorized_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const float4*)conv_out.data_ptr<float>(),
        (const float4*)add_input.data_ptr<float>(),
        (float4*)output.data_ptr<float>(),
        numel_v4
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "CUDA kernel failed: ", cudaGetErrorString(cudaGetLastError()));
}
"""

# --- C++ Binding ---
cpp_source = r"""
#include <torch/extension.h>

void launch_fused_add_hardswish(
    const at::Tensor& conv_out,
    const at::Tensor& add_input,
    at::Tensor& output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_add_hardswish", &launch_fused_add_hardswish, "Vectorized Fused Add + HardSwish");
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

# --- Custom ConvTranspose3D CUDA Kernel ---
conv_transpose3d_cuda = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int Ci, int Di, int Hi, int Wi,
    int Co, int K, int S, int P, int OP, int Dout, int Hout, int Wout) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * Co * Dout * Hout * Wout;
    if (idx >= total_elements) return;
    
    int n = idx / (Co * Dout * Hout * Wout);
    int temp = idx % (Co * Dout * Hout * Wout);
    int co = temp / (Dout * Hout * Wout);
    temp = temp % (Dout * Hout * Wout);
    int dout = temp / (Hout * Wout);
    temp = temp % (Hout * Wout);
    int hout = temp / Wout;
    int wout = temp % Wout;
    
    float sum = 0.0f;
    for (int ci = 0; ci < Ci; ci++) {
        for (int kd = 0; kd < K; kd++) {
            for (int kh = 0; kh < K; kh++) {
                for (int kw = 0; kw < K; kw++) {
                    int din = (dout + OP - kd) / S;
                    int hin = (hout + OP - kh) / S;
                    int win = (wout + OP - kw) / S;
                    
                    if ((dout + OP - kd) % S == 0 && 
                        (hout + OP - kh) % S == 0 && 
                        (wout + OP - kw) % S == 0 &&
                        din >= 0 && din < Di &&
                        hin >= 0 && hin < Hi &&
                        win >= 0 && win < Wi) {
                        int input_idx = n * Ci * Di * Hi * Wi + 
                                        ci * Di * Hi * Wi + 
                                        din * Hi * Wi + 
                                        hin * Wi + win;
                        int weight_idx = co * Ci * K * K * K + 
                                         ci * K * K * K + 
                                         (K - 1 - kd) * K * K + 
                                         (K - 1 - kh) * K + 
                                         (K - 1 - kw);
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    output[idx] = sum + bias[co];
}

at::Tensor conv_transpose3d_cuda_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    int stride, int padding, int output_padding) {
    
    int N = input.size(0);
    int Ci = input.size(1);
    int Di = input.size(2);
    int Hi = input.size(3);
    int Wi = input.size(4);
    
    int Co = weight.size(0);
    int K = weight.size(2);
    
    int Dout = (Di - 1) * stride - 2 * padding + K + output_padding;
    int Hout = (Hi - 1) * stride - 2 * padding + K + output_padding;
    int Wout = (Wi - 1) * stride - 2 * padding + K + output_padding;
    
    auto output = at::zeros({N, Co, Dout, Hout, Wout}, input.options());
    
    const dim3 block(256);
    const int total_elements = N * Co * Dout * Hout * Wout;
    const dim3 grid((total_elements + block.x - 1) / block.x);
    
    conv_transpose3d_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, Ci, Di, Hi, Wi,
        Co, K, stride, padding, output_padding, Dout, Hout, Wout
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "ConvTranspose3D kernel failed: ", cudaGetErrorString(cudaGetLastError()));
    return output;
}
"""

conv_transpose3d_cpp = r"""
#include <torch/extension.h>

at::Tensor conv_transpose3d_cuda_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    int stride, int padding, int output_padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose3d", &conv_transpose3d_cuda_impl, "ConvTranspose3D CUDA Implementation");
}
"""

conv_transpose_ext = load_inline(
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
    x = conv_transpose_ext.conv_transpose3d(
        x, conv_transpose_weight, conv_transpose_bias,
        conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding
    )
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Launch fused CUDA kernel for element-wise operations
    fused_ext.fused_add_hardswish(x, add_input, output)
    
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
