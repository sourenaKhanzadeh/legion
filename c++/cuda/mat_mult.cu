#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// CUDA Kernel for Matrix Multiplication (Supports Any MxN * NxK)
__global__ void matrixMulKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * K + col];
        }
        C[row * K + col] = sum;
    }
}

// CUDA Function for Matrix Multiplication
torch::Tensor matrixMultiply(torch::Tensor A, torch::Tensor B) {
    // Ensure A and B are contiguous & on GPU
    A = A.contiguous().to(torch::kCUDA);
    B = B.contiguous().to(torch::kCUDA);

    // Get matrix dimensions
    int M = A.size(0);  // Rows of A
    int N = A.size(1);  // Columns of A / Rows of B
    int K = B.size(1);  // Columns of B

    // Output matrix C on CUDA
    torch::Tensor C = torch::zeros({M, K}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // CUDA block & grid configuration
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch CUDA Kernel
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                                    C.data_ptr<float>(), M, N, K);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA Kernel Launch Error: ") + cudaGetErrorString(err));
    }

    // Synchronize CUDA with PyTorch
    torch::cuda::synchronize();

    return C;
}

// Pybind11 binding
PYBIND11_MODULE(mat_mult, m) {
    m.def("matrix_multiply", &matrixMultiply, "Matrix multiplication using CUDA");
}
