# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_134650/code_15.py
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

# --------------------------------------------------------------
#  CUDA kernel with coalesced memory accesses + loop unrolling
# --------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_dim1_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int B, int D1, int D2) {
    // --- block / thread indexing ---------------------------------
    int b = blockIdx.x;
    int j = (blockIdx.y * blockDim.x + threadIdx.x) * 4; // 4 output elements per thread

    if (b < B && j < D2) {
        // --- how many valid lanes (1‑4) does this thread have? ----
        int vec_len = (j + 4 <= D2) ? 4 : (D2 - j);

        // --- accumulators -----------------------------------------
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

        // --- base offset for the batch -----------------------------
        int base = b * (D1 * D2) + j;

        // --- unrolled reduction loop (D1 = 4096) -------------------
        // Unroll by 8 → 4096 / 8 = 512 iterations
        #pragma unroll 8
        for (int i = 0; i < D1; ++i) {
            // The vector loop is fully unrolled; the if‑statements are
            // compile‑time constants for a given thread, so they become
            // predicated instructions rather than branches.
            if (vec_len > 0) sum0 += input[base + i * D2];
            if (vec_len > 1) sum1 += input[base + i * D2 + 1];
            if (vec_len > 2) sum2 += input[base + i * D2 + 2];
            if (vec_len > 3) sum3 += input[base + i * D2 + 3];
        }

        // --- store the result --------------------------------------
        output[b * D2 + j] = sum0;
        if (vec_len > 1) output[b * D2 + j + 1] = sum1;
        if (vec_len > 2) output[b * D2 + j + 2] = sum2;
        if (vec_len > 3) output[b * D2 + j + 3] = sum3;
    }
}

void sum_dim1(torch::Tensor input, torch::Tensor output) {
    const int B  = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);

    // 256 threads per block gives good occupancy on RTX 2080 Ti
    dim3 threads(256);
    // Grid in y dimension covers the output width (D2) with vector width = 4
    dim3 blocks(B, (D2 + 4 * threads.x - 1) / (4 * threads.x));

    sum_dim1_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), B, D1, D2);
}
"""

# --------------------------------------------------------------
#  C++ binding (PYBIND11)
# --------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void sum_dim1(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1", &sum_dim1, "Sum along dim=1 with coalesced memory and loop unrolling");
}
"""

# --------------------------------------------------------------
#  Compile the extension
# --------------------------------------------------------------
sum_ext = load_inline(
    name='sum_dim1',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# --------------------------------------------------------------
#  Functional wrapper required by the evaluation harness
# --------------------------------------------------------------
def functional_model(x, *, dim):
    """Sum tensor x along the given dimension (only dim==1 is supported)."""
    assert dim == 1
    # Output shape: (batch_size, 1, dim2)
    out = torch.empty((x.shape[0], x.shape[2]), device=x.device, dtype=x.dtype)
    sum_ext.sum_dim1(x, out)
    return out.unsqueeze(1)


# --------------------------------------------------------------
#  Quick sanity‑check (can be removed in production)
# --------------------------------------------------------------
if __name__ == "__main__":
    batch, d1, d2 = 8, 128, 127
    x = torch.rand(batch, d1, d2, device='cuda')
    y = functional_model(x, dim=1)
    # Compare with pure PyTorch result (within floating‑point tolerance)
    y_ref = x.sum(dim=1, keepdim=True)
    assert torch.allclose(y, y_ref, atol=1e-5)
    print("Result matches PyTorch reference.")
