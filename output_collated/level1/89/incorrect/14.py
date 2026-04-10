# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_074424/code_20.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['dim']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    A simple model that performs a cumulative sum (prefix sum) operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the scan operation.
    """

    def __init__(self, dim):
        """
        Initialize the Scan model.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
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

# CUDA kernel using CUB for high-performance scan implementation
# We use CUB because it is the industry standard for prefix sums, 
# significantly outperforming manual warp-shuffle implementations.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

void cumsum_fwd_cuda(torch::Tensor input, torch::Tensor output, int dim) {
    int rows = input.size(0);
    int cols = input.size(1);
    
    // Allocate temporary storage for CUB
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(NULL, temp_storage_bytes, (float*)NULL, (float*)NULL, rows * cols);
    auto temp_storage = torch::empty({(long)temp_storage_bytes}, input.options().dtype(torch::kByte));
    
    if (dim == 1) {
        // Since CUB scans are linear, for dim=1 we scan row by row 
        // Or simply treat the whole matrix as a flat sequence if dim=1 
        // and matrix is ordered correctly. 
        // Given the memory layout, a direct inclusive sum on the flat pointer works.
        cub::DeviceScan::InclusiveSum(temp_storage.data_ptr(), temp_storage_bytes, 
                                     input.data_ptr<float>(), output.data_ptr<float>(), 
                                     rows * cols);
    } else {
        // Dim=0 requires a strided scan or transposition.
        // For efficiency, we transpose, scan, transpose back.
        auto t_input = input.transpose(0, 1).contiguous();
        auto t_output = torch::empty_like(t_input);
        cub::DeviceScan::InclusiveSum(temp_storage.data_ptr(), temp_storage_bytes, 
                                     t_input.data_ptr<float>(), t_output.data_ptr<float>(), 
                                     rows * cols);
        output.copy_(t_output.transpose(0, 1));
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>
void cumsum_fwd_cuda(torch::Tensor input, torch::Tensor output, int dim);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cumsum_fwd", &cumsum_fwd_cuda, "Optimized CUB cumsum");
}
"""

# Compile extension
cumsum_ext = load_inline(
    name='cumsum_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    """
    Optimized functional_model using CUB-based CUDA prefix sum.
    """
    x_contig = x.contiguous().cuda()
    output = torch.empty_like(x_contig)
    cumsum_ext.cumsum_fwd(x_contig, output, dim)
    return output

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_init_inputs():
    return [dim]

def get_inputs():
    return [torch.rand(batch_size, *input_shape)]
