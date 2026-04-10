# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_123707/code_12.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['dim']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs sum reduction over a specified dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

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
    if 'dim' in flat_state:
        state_kwargs['dim'] = flat_state['dim']
    else:
        state_kwargs['dim'] = getattr(model, 'dim')
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

// Each block reduces a chunk of the dim_size (4096).
// We launch blocks = outer * inner.
__global__ void sum_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int outer,
    const int reduce_dim,
    const int inner
) {
    // Mapping: blockIdx.x corresponds to a flattened (outer, inner) index 
    // where inner is innermost.
    int idx = blockIdx.x;
    int outer_idx = idx / inner;
    int inner_idx = idx % inner;
    
    // Pointer to the start of the vector to be reduced
    const float* input_ptr = input + (outer_idx * reduce_dim * inner) + inner_idx;
    
    extern __shared__ float sdata[];
    float thread_sum = 0.0f;
    
    // Coalesced loading via strided access: stride is 'inner'
    for (int i = threadIdx.x; i < reduce_dim; i += blockDim.x) {
        thread_sum += input_ptr[i * inner];
    }
    
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();
    
    // Hierarchical reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        output[idx] = sdata[0];
    }
}

void sum_reduce_launch(const at::Tensor& input, at::Tensor& output, int dim) {
    int outer = 1;
    for(int i = 0; i < dim; ++i) outer *= input.size(i);
    int reduce_dim = input.size(dim);
    int inner = 1;
    for(int i = dim + 1; i < input.dim(); ++i) inner *= input.size(i);
    
    int threads = 256;
    int blocks = outer * inner;
    
    sum_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), outer, reduce_dim, inner
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void sum_reduce_launch(const at::Tensor& input, at::Tensor& output, int dim);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_reduce", &sum_reduce_launch, "Sum reduce");
}
"""

sum_ext = load_inline(
    name='sum_reduce_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3'],
    with_cuda=True
)

def functional_model(x, *, dim):
    # Prepare output shape: keepdim is implicitly True by output_shape logic
    output_shape = list(x.shape)
    output_shape[dim] = 1
    output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    sum_ext.sum_reduce(x, output, dim)
    return output

# Inputs as per requested interface
batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def get_init_inputs():
    return [reduce_dim]

def get_inputs():
    return [torch.rand(batch_size, dim1, dim2, device='cuda')]
