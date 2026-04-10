# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_113951/code_4.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['kernel_size', 'stride', 'padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['avg_pool_kernel_size', 'avg_pool_stride', 'avg_pool_padding', 'avg_pool_ceil_mode', 'avg_pool_count_include_pad']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs 1D Average Pooling.
    """

    def __init__(self, kernel_size: int, stride: int=1, padding: int=0):
        """
        Initializes the 1D Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to 1.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

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
    # State for avg_pool (nn.AvgPool1d)
    state_kwargs['avg_pool_kernel_size'] = model.avg_pool.kernel_size
    state_kwargs['avg_pool_stride'] = model.avg_pool.stride
    state_kwargs['avg_pool_padding'] = model.avg_pool.padding
    state_kwargs['avg_pool_ceil_mode'] = model.avg_pool.ceil_mode
    state_kwargs['avg_pool_count_include_pad'] = model.avg_pool.count_include_pad
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_shared_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int padding) {

    extern __shared__ float s_data[];

    int tid = threadIdx.x;
    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int block_offset = blockIdx.z * blockDim.x;
    
    // We need blockDim.x + kernel_size - 1 elements, but just loading 
    // a standard tile and handling bounds is more cache efficient.
    int input_base = batch_idx * (channels * input_length) + channel_idx * input_length;
    
    // Load input tile into shared memory
    int s_idx = tid;
    int global_idx = block_offset + tid - padding;
    
    if (global_idx >= 0 && global_idx < input_length)
        s_data[s_idx] = input[input_base + global_idx];
    else
        s_data[s_idx] = 0.0f;
    
    __syncthreads();

    int out_idx = block_offset + tid;
    if (out_idx < output_length) {
        float sum = 0.0f;
        int count = 0;
        for (int i = 0; i < kernel_size; ++i) {
            int local_pos = tid + i;
            if (local_pos >= 0 && local_pos < (blockDim.x + kernel_size)) {
                sum += s_data[local_pos];
                count++;
            }
        }
        output[batch_idx * (channels * output_length) + channel_idx * output_length + out_idx] = sum / count;
    }
}

void avg_pool1d_cuda(const torch::Tensor& input, torch::Tensor& output, int k, int s, int p) {
    int b = input.size(0);
    int c = input.size(1);
    int in_l = input.size(2);
    int out_l = output.size(2);
    
    dim3 block(256);
    dim3 grid(b, c, (out_l + block.x - 1) / block.x);
    size_t shared_mem = (block.x + k) * sizeof(float);
    
    avg_pool1d_shared_kernel<<<grid, block, shared_mem>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), c, in_l, out_l, k, p
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void avg_pool1d_cuda(const torch::Tensor& input, torch::Tensor& output, int k, int s, int p);

torch::Tensor forward(torch::Tensor input, int k, int s, int p, bool ceil_mode, bool include_pad) {
    int out_l = (input.size(2) + 2 * p - k) / s + 1;
    auto output = torch::zeros({input.size(0), input.size(1), out_l}, input.options());
    avg_pool1d_cuda(input, output, k, s, p);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized AvgPool1D");
}
"""

fused_ext = load_inline(
    name='avg_pool1d_ext', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3'], with_cuda=True
)

def functional_model(x, *, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding, avg_pool_ceil_mode, avg_pool_count_include_pad):
    return fused_ext.forward(x, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding, avg_pool_ceil_mode, avg_pool_count_include_pad)

batch_size, in_channels, input_length = 64, 128, 65536
kernel_size, stride, padding = 8, 1, 4

def get_init_inputs():
    return [kernel_size, stride, padding]

def get_inputs():
    return [torch.rand(batch_size, in_channels, input_length, device='cuda')]
