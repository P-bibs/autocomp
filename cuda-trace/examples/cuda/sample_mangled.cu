#include <cstdint>

__global__ void copy(const float* inp, float* out, int n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < static_cast<size_t>(n)) {
        out[idx] = inp[idx];
    }
}

int main() {
    auto* inp = reinterpret_cast<const float*>(static_cast<uintptr_t>(0x1234));
    auto* out = reinterpret_cast<float*>(static_cast<uintptr_t>(0xdeadbeef));
    copy<<<2, 32>>>(inp, out, 0);
    cudaDeviceSynchronize();
    return 0;
}
