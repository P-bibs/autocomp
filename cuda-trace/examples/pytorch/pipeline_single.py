import torch
from torch import nn
from torch.utils.cpp_extension import load_inline


cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void scale_kernel(const T* input, T* output, const T scale, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * scale;
    }
}

torch::Tensor scale_forward(torch::Tensor input, double scale) {
    auto output = torch::empty_like(input);
    const size_t size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "scale_kernel", [&] {
        scale_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            static_cast<scalar_t>(scale),
            size
        );
    });
    cudaDeviceSynchronize();
    return output;
}
"""


cpp_source = r"""
#include <torch/extension.h>

torch::Tensor scale_forward(torch::Tensor input, double scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &scale_forward, "Scale operation");
}
"""


scale_mod = load_inline(
    name="pipeline_scale_single",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
)


def get_init_inputs():
    in_dim = 128
    out_dim = 128
    scale = 3.0
    return (in_dim, out_dim, scale)


def get_inputs():
    batch_size = 32
    in_dim = 128
    return (torch.rand(batch_size, in_dim).cuda(),)


class ModelNew(nn.Module):
    def __init__(self, in_dim, out_dim, scale):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.scale = float(scale)

    def forward(self, x):
        a = self.linear(x)
        b = scale_mod.forward(a, self.scale)
        return b
