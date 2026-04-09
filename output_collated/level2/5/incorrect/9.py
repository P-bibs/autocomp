# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_112831/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'bias_shape', 'stride', 'padding', 'output_padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
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
    # State for conv_transpose (nn.ConvTranspose2d)
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

# --- CUDA Kernel ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_SIZE 16
#define KERNEL_SIZE 4

__global__ void fused_conv_transpose_tanh_kernel(
    const float* __restrict__ input,         // shape: [B, C_in, H_in, W_in]
    const float* __restrict__ weight,        // shape: [C_in, C_out, K, K]
    const float* __restrict__ conv_bias,     // shape: [C_out]
    const float* __restrict__ tanh_bias,     // shape: [C_out, 1, 1]
    float* __restrict__ output,              // shape: [B, C_out, H_out, W_out]
    int B, int C_in, int C_out,
    int H_in, int W_in,
    int H_out, int W_out
) {
    // Each block handles one output channel & spatial tile
    int batch_id = blockIdx.x;
    int out_ch = blockIdx.y;
    int tile_x = blockIdx.z % (W_out / TILE_SIZE);
    int tile_y = blockIdx.z / (W_out / TILE_SIZE);

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Shared memory for input tile and weight
    __shared__ float s_input[TILE_SIZE + 3][TILE_SIZE + 3]; // +3 for 4x4 kernel padding
    __shared__ float s_weight[KERNEL_SIZE][KERNEL_SIZE];

    float acc = 0.0f;

    // Loop over input channels
    for (int c_in = 0; c_in < C_in; ++c_in) {
        // Load input tile into shared memory
        int x = tile_x * TILE_SIZE + tx;
        int y = tile_y * TILE_SIZE + ty;

        if (x < W_in && y < H_in) {
            s_input[ty][tx] = input[((batch_id * C_in + c_in) * H_in + y) * W_in + x];
        } else {
            s_input[ty][tx] = 0.0f;
        }

        // Load weight into shared memory
        if (tx < KERNEL_SIZE && ty < KERNEL_SIZE) {
            s_weight[ty][tx] = weight[((c_in * C_out + out_ch) * KERNEL_SIZE + ty) * KERNEL_SIZE + tx];
        }

        __syncthreads();

        // Perform transposed convolution sum with stride=2
        for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
            for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                int out_x = x * 2 + kx; // assuming stride 2
                int out_y = y * 2 + ky;

                if (out_x < W_out && out_y < H_out) {
                    acc += s_input[ty][tx] * s_weight[ky][kx];
                }
            }
        }
        __syncthreads();
    }

    // Apply bias and tanh
    int out_x_idx = tile_x * TILE_SIZE + tx;
    int out_y_idx = tile_y * TILE_SIZE + ty;

    if (out_x_idx < W_out && out_y_idx < H_out) {
        float val = acc + conv_bias[out_ch];
        val -= tanh_bias[out_ch]; // fused bias subtraction
        val = tanhf(val);         // fused tanh
        output[((batch_id * C_out + out_ch) * H_out + out_y_idx) * W_out + out_x_idx] = val;
    }
}

void launch_fused_op(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const torch::Tensor &conv_bias,
    const torch::Tensor &tanh_bias,
    torch::Tensor &output
) {
    int B = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);

    int C_out = weight.size(1);
    int H_out = H_in * 2; // assuming stride = 2
    int W_out = W_in * 2;

    int grid_x = B;
    int grid_y = C_out;
    int grid_z = (H_out / TILE_SIZE) * (W_out / TILE_SIZE);
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(TILE_SIZE, TILE_SIZE);

    fused_conv_transpose_tanh_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        tanh_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C_in, C_out, H_in, W_in, H_out, W_out
    );
}
"""

# --- C++ Binding ---
cpp_source = r"""
#include <torch/extension.h>

void launch_fused_op(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const torch::Tensor &conv_bias,
    const torch::Tensor &tanh_bias,
    torch::Tensor &output
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_fused_op, "Fused Conv Transpose + Bias Sub + Tanh");
}
"""

# Compile the extension with optimizations
fused_ext = load_inline(
    name='fused_op_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# Rewritten functional_model using fused CUDA kernel
def functional_model(
    x,
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
    # Only stride = 2 is supported in this kernel for simplicity
    assert conv_transpose_stride == (2, 2)
    out_tensor = torch.empty(
        x.shape[0],
        conv_transpose_weight.shape[1],
        x.shape[2] * 2,
        x.shape[3] * 2,
        device=x.device,
        dtype=x.dtype
    )
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, bias, out_tensor)
    return out_tensor

batch_size = 32
in_channels  = 64  
out_channels = 64  
height = width = 256 
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
