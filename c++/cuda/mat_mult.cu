#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

__global__ void matrixMulKernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matrixMultiply(torch::Tensor A, torch::Tensor B, int N) {
    A = A.to(torch::kCUDA);
    B = B.to(torch::kCUDA);
    torch::Tensor C = torch::zeros({N, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    cudaDeviceSynchronize();

    return C;
}

// Pybind11 binding
PYBIND11_MODULE(mat_mult, m) {
    m.def("matrix_multiply", &matrixMultiply, "Matrix multiplication using CUDA");
}
