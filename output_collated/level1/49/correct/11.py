# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_152959/code_6.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['dim']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs Max reduction over a specific dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
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

# ------------------------------------------------------------------
#  CUDA kernel source (max reduction)
# ------------------------------------------------------------------
cuda_kernel = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

extern "C" __global__
void max_reduce_kernel(const float* __restrict__ src,
                       float* __restrict__ dst,
                       const int N,          // outer size before reduction
                       const int K,          // reduction size
                       const int M) {        // inner size after reduction
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const int total = N * M;

    for (int i = idx; i < total; i += stride) {
        int n = i / M;
        int m = i % M;

        // Pointer arithmetic for specific layout: [N][K][M]
        // M is the fastest varying dimension.
        const float* slice = src + n * K * M + m;

        float best = -FLT_MAX;
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float v = slice[k * M];
            if (v > best) {
                best = v;
            }
        }
        dst[n * M + m] = best;
    }
}

void max_reduce_launcher(const torch::Tensor& src,
                         torch::Tensor& dst,
                         const int dim) {
    const auto shape = src.sizes();
    int ndim = shape.size();

    int N = 1;
    for (int i = 0; i < dim; ++i) N *= shape[i];
    int K = shape[dim];
    int M = 1;
    for (int i = dim + 1; i < ndim; ++i) M *= shape[i];

    const int threads = 256;
    const int output_elements = N * M;
    const int blocks = std::min((output_elements + threads - 1) / threads, 4096);

    max_reduce_kernel<<<blocks, threads>>>(
        src.data_ptr<float>(), 
        dst.data_ptr<float>(), 
        N, K, M
    );
}
'''

# ------------------------------------------------------------------
#  C++ binding
# ------------------------------------------------------------------
cpp_source = r'''
#include <torch/extension.h>

void max_reduce_launcher(const torch::Tensor& src, torch::Tensor& dst, const int dim);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reduce", &max_reduce_launcher, "Maximum reduction along a single dimension (CUDA)");
}
'''

# ------------------------------------------------------------------
#  Build extension
# ------------------------------------------------------------------
fused_ext = load_inline(
    name='max_reduce_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False,
)

# ------------------------------------------------------------------
#  Functional model
# ------------------------------------------------------------------
def functional_model(x, *, dim):
    # Ensure tensor is contiguous and float32
    if not x.is_contiguous():
        x = x.contiguous()
    
    out_shape = list(x.shape)
    del out_shape[dim]
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    fused_ext.max_reduce(x, out, dim)
    return out

# Helper functions provided for consistency
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_init_inputs():
    return [1]

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device='cuda', dtype=torch.float32)
    return [x]
