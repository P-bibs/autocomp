# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_040909/code_2.py
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

# --- CUDA Kernel for Fused Operation with Vectorization ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

typedef float4 vec_t;

__device__ __forceinline__ float4 hardswish_vec(float4 x) {
    float4 res;
    #pragma unroll
    for(int i=0; i<4; ++i) {
        float val = ((float*)&x)[i];
        float relu6_val = fminf(fmaxf(val + 3.0f, 0.0f), 6.0f);
        ((float*)&res)[i] = val * relu6_val * 0.16666667f; // 1/6
    }
    return res;
}

__global__ void fused_conv_transpose3d_add_hardswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_D, const int input_H, const int input_W,
    const int output_D, const int output_H, const int output_W,
    const int kernel_size,
    const int stride,
    const int padding) {

    const int out_C = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (out_C >= out_channels || idx >= batch_size * output_D * output_H * output_W) return;

    const int w = idx % output_W;
    const int h = (idx / output_W) % output_H;
    const int d = (idx / (output_W * output_H)) % output_D;
    const int b = idx / (output_W * output_H * output_D);

    float sum = 0.0f;

    const int in_C_start = (out_C / (out_channels / in_channels));
    const int in_C_end = in_C_start + 1;

    for (int in_c = in_C_start; in_c < in_C_end; ++in_c) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            const int in_d = d + padding - kd * stride;
            if (in_d % stride != 0) continue;
            const int src_d = in_d / stride;
            if (src_d < 0 || src_d >= input_D) continue;

            for (int kh = 0; kh < kernel_size; ++kh) {
                const int in_h = h + padding - kh * stride;
                if (in_h % stride != 0) continue;
                const int src_h = in_h / stride;
                if (src_h < 0 || src_h >= input_H) continue;

                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int in_w = w + padding - kw * stride;
                    if (in_w % stride != 0) continue;
                    const int src_w = in_w / stride;
                    if (src_w < 0 || src_w >= input_W) continue;

                    const int input_idx = b * (in_channels * input_D * input_H * input_W) +
                                          in_c * (input_D * input_H * input_W) +
                                          src_d * (input_H * input_W) +
                                          src_h * input_W +
                                          src_w;
                    const int weight_idx = out_C * (in_channels * kernel_size * kernel_size * kernel_size) +
                                           in_c * (kernel_size * kernel_size * kernel_size) +
                                           kd * (kernel_size * kernel_size) +
                                           kh * kernel_size +
                                           kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    sum += bias[out_C];
    
    const int out_idx = b * (out_channels * output_D * output_H * output_W) +
                        out_C * (output_D * output_H * output_W) +
                        d * (output_H * output_W) +
                        h * output_W +
                        w;

    float x = sum + add_input[out_idx];
    float relu6_val = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    output[out_idx] = x * relu6_val * 0.16666667f;
}

void launch_fused_conv_transpose3d_add_hardswish(
    const at::Tensor& input,
    const at::Tensor& add_input,
    at::Tensor& output,
    const at::Tensor& weight,
    const at::Tensor& bias,
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
    
    const dim3 block(16, 16);
    const dim3 grid((out_channels + block.x - 1) / block.x, 
                    (batch_size * output_D * output_H * output_W + block.y - 1) / block.y);
    
    fused_conv_transpose3d_add_hardswish_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        add_input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
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

void launch_fused_conv_transpose3d_add_hardswish(
    const at::Tensor& input,
    const at::Tensor& add_input,
    at::Tensor& output,
    const at::Tensor& weight,
    const at::Tensor& bias,
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
    # Allocate output tensor
    output = torch.empty(
        x.size(0),
        conv_transpose_weight.size(0),
        (x.size(2) - 1) * conv_transpose_stride + conv_transpose_dilation * (conv_transpose_weight.size(2) - 1) + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding,
        (x.size(3) - 1) * conv_transpose_stride + conv_transpose_dilation * (conv_transpose_weight.size(3) - 1) + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding,
        (x.size(4) - 1) * conv_transpose_stride + conv_transpose_dilation * (conv_transpose_weight.size(4) - 1) + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding,
        device=x.device,
        dtype=x.dtype
    )
    
    # Launch fused CUDA kernel for all operations
    fused_ext.fused_conv_transpose3d_add_hardswish(
        x, add_input, output, 
        conv_transpose_weight, conv_transpose_bias,
        conv_transpose_weight.size(2),  # kernel_size (assuming cubic)
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

