extern "C" __global__ void copy_kernel(const float* inp, float* out, int n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < static_cast<size_t>(n)) {
        out[idx] = inp[idx];
    }
}

int main() {
    copy_kernel<<<1, 1>>>(nullptr, nullptr, 0);
    cudaDeviceSynchronize();
    return 0;
}
