import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_kernel = r"""
#include <torch/extension.h>
#include <cmath>

__global__ void fused_op_kernel(float* __restrict__ x,
                                const float sub,
                                const float mul,
                                const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = fmaxf((x[idx] - sub) * mul, 0.0f);
    }
}

void fused_op(torch::Tensor x, float sub, float mul) {
    const int size = x.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_op_kernel", [&] {
        fused_op_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(),
            sub,
            mul,
            size
        );
    });
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op(torch::Tensor x, float sub, float mul);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused subtract/multiply/ReLU");
}
"""

fused_op_lib = load_inline(
    name="fused_op_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, sub, mul):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.sub = sub
        self.mul = mul

    def forward(self, x):
        x = self.linear(x)
        fused_op_lib.fused_op(x, self.sub, self.mul)
        return x
# /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260407_002851/code_10.py
