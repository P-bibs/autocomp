# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_075452/code_5.py
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

# Optimization: Convert cumsum to a custom CUDA kernel using work-efficient parallel scan
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__ void work_efficient_scan_kernel(const float* input, float* output, int batch_size, int seq_len) {
    typedef cub::BlockScan<float, 1024> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // Process one batch element per block
    if (batch_idx >= batch_size) return;
    
    const float* row_in = input + batch_idx * seq_len;
    float* row_out = output + batch_idx * seq_len;
    
    // Perform segmented scan on the sequence
    for (int offset = 0; offset < seq_len; offset += blockDim.x) {
        int idx = offset + tid;
        float thread_data = (idx < seq_len) ? row_in[idx] : 0.0f;
        
        // Perform inclusive scan
        BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);
        __syncthreads();
        
        if (idx < seq_len) {
            row_out[idx] = thread_data;
        }
    }
}

void scan_cuda(torch::Tensor input, torch::Tensor output) {
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int threads = 1024;
    int blocks = batch_size;
    
    work_efficient_scan_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        seq_len
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void scan_cuda(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scan", &scan_cuda, "Work-efficient parallel prefix sum");
}
"""

# Compile the extension
scan_ext = load_inline(
    name='scan_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    # Only support dim=1 for this implementation as per input spec
    if dim != 1:
        raise ValueError("Only dim=1 is supported")
    
    # Ensure input is on GPU
    if not x.is_cuda:
        x = x.cuda()
    
    out = torch.empty_like(x)
    scan_ext.scan(x, out)
    return out

# Initialization parameters
batch_size = 32768
input_shape = (32768,)
dim = 1

def get_init_inputs():
    return [dim]

def get_inputs():
    return [torch.rand(batch_size, *input_shape)]
