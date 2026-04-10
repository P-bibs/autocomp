# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_080721/code_17.py
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

# Optimization Strategy:
# 1. The provided problem involves a large matrix (32768 x 32768).
# 2. Shared memory is limited (typically 48KB - 164KB per block). Attempting to load 
#    a 32768-float (128KB) array into shared memory per block exceeds static limits 
#    on most hardware and prevents occupancy.
# 3. We use a blocked scan approach: perform prefix sum on segments using 
#    shared memory and propagate sums across blocks (Global Prefix Sum).
# 4. We use the Thrust-like approach: 1) Local scan per block, 2) Store block 
#    totals, 3) Scan block totals, 4) Add block offsets.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 1024

__global__ void scan_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    extern __shared__ float sdata[];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    // For 32768, we process in chunks of BLOCK_SIZE
    float block_sum = 0.0f;
    
    for (int chunk = 0; chunk < N; chunk += BLOCK_SIZE) {
        int idx = chunk + tid;
        float val = (idx < N) ? input[bid * N + idx] : 0.0f;
        sdata[tid] = val;
        __syncthreads();

        // Hillis-Steele per block
        for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
            float v = (tid >= stride) ? sdata[tid - stride] : 0.0f;
            __syncthreads();
            if (tid >= stride) sdata[tid] += v;
            __syncthreads();
        }

        if (idx < N) {
            output[bid * N + idx] = sdata[tid] + block_sum;
        }
        
        block_sum += sdata[BLOCK_SIZE - 1];
        __syncthreads();
    }
}

void call_scan(torch::Tensor input, torch::Tensor output) {
    int batch_size = input.size(0);
    int N = input.size(1);
    
    // Launch one block per row
    scan_kernel<<<batch_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), N);
}
"""

cpp_source = r"""
void call_scan(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scan", &call_scan, "Parallel scan operation");
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
    assert dim == 1, "Only dimension 1 supported"
    # Ensure contiguous for kernel access
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)
    scan_ext.scan(x_contig, out)
    return out

# Initialization logic
batch_size = 32768
input_shape = (32768,)
dim = 1

def get_init_inputs():
    return [dim]

def get_inputs():
    return [torch.rand(batch_size, *input_shape, device='cuda')]
