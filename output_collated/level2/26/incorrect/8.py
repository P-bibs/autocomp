# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_040909/code_0.py
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

# --- CUDA Kernel for Fused ConvTranspose3D + Add + HardSwish ---
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
    const float* __restrict__ bias,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w) {

    // Calculate output indices
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    
    if (tid >= total_elements) return;
    
    int w_out = tid % output_w;
    int h_out = (tid / output_w) % output_h;
    int d_out = (tid / (output_w * output_h)) % output_d;
    int c_out = (tid / (output_w * output_h * output_d)) % out_channels;
    int n = tid / (output_w * output_h * output_d * out_channels);
    
    // Calculate input coordinates
    float sum = (bias != nullptr) ? bias[c_out] : 0.0f;
    
    // Conv transpose calculation
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    // Calculate corresponding input position
                    int in_d = d_out + padding_d - kd * dilation_d;
                    int in_h = h_out + padding_h - kh * dilation_h;
                    int in_w = w_out + padding_w - kw * dilation_w;
                    
                    // Check if within input bounds and stride alignment
                    if (in_d >= 0 && in_d < input_d * stride_d && in_d % stride_d == 0 &&
                        in_h >= 0 && in_h < input_h * stride_h && in_h % stride_h == 0 &&
                        in_w >= 0 && in_w < input_w * stride_w && in_w % stride_w == 0) {
                        
                        int orig_d = in_d / stride_d;
                        int orig_h = in_h / stride_h;
                        int orig_w = in_w / stride_w;
                        
                        if (orig_d < input_d && orig_h < input_h && orig_w < input_w) {
                            int input_idx = n * (in_channels * input_d * input_h * input_w) +
                                          c_in * (input_d * input_h * input_w) +
                                          orig_d * (input_h * input_w) +
                                          orig_h * input_w +
                                          orig_w;
                          
                            int weight_idx = c_out * (in_channels * kernel_d * kernel_h * kernel_w) +
                                           c_in * (kernel_d * kernel_h * kernel_w) +
                                           kd * (kernel_h * kernel_w) +
                                           kh * kernel_w +
                                           kw;
                                           
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add element from add_input and apply hardswish
    sum += add_input[tid];
    output[tid] = hardswish_impl(sum);
}

void launch_fused_conv_transpose3d_add_hardswish(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& add_input,
    at::Tensor& output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_d = input.size(2);
    int input_h = input.size(3);
    int input_w = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);
    
    int output_d = output.size(2);
    int output_h = output.size(3);
    int output_w = output.size(4);
    
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    const dim3 block(256);
    const dim3 grid((total_elements + block.x - 1) / block.x);
    
    fused_conv_transpose3d_add_hardswish_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        add_input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w
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
    const at::Tensor& bias,
    const at::Tensor& add_input,
    at::Tensor& output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w);

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
    # Calculate output dimensions for conv transpose
    stride = conv_transpose_stride if isinstance(conv_transpose_stride, (list, tuple)) else (conv_transpose_stride, conv_transpose_stride, conv_transpose_stride)
    padding = conv_transpose_padding if isinstance(conv_transpose_padding, (list, tuple)) else (conv_transpose_padding, conv_transpose_padding, conv_transpose_padding)
    dilation = conv_transpose_dilation if isinstance(conv_transpose_dilation, (list, tuple)) else (conv_transpose_dilation, conv_transpose_dilation, conv_transpose_dilation)
    
    if len(stride) == 1:
        stride = (stride[0], stride[0], stride[0])
    if len(padding) == 1:
        padding = (padding[0], padding[0], padding[0])
    if len(dilation) == 1:
        dilation = (dilation[0], dilation[0], dilation[0])
    
    # Output dimensions calculation for conv transpose
    out_d = (x.shape[2] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (conv_transpose_weight.shape[2] - 1) + 1 + conv_transpose_output_padding
    out_h = (x.shape[3] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (conv_transpose_weight.shape[3] - 1) + 1 + conv_transpose_output_padding
    out_w = (x.shape[4] - 1) * stride[2] - 2 * padding[2] + dilation[2] * (conv_transpose_weight.shape[4] - 1) + 1 + conv_transpose_output_padding
    
    # Allocate output tensor
    output = torch.empty(x.shape[0], conv_transpose_weight.shape[0], out_d, out_h, out_w, device=x.device, dtype=x.dtype)
    
    # Launch fused CUDA kernel for all operations
    fused_ext.fused_conv_transpose3d_add_hardswish(
        x, conv_transpose_weight, conv_transpose_bias, add_input, output,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2]
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
