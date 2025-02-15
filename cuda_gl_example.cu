#include <iostream>
#include <glm/glm.hpp> // Include GLM for vec2
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                         \
    do                                                                           \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess)                                                  \
        {                                                                        \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;                   \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

// CUDA kernel to add two glm::vec2 vectors
__global__ void addVectors(const glm::vec2 *a, const glm::vec2 *b, glm::vec2 *result, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        result[idx] = a[idx] + b[idx]; // GLM vector addition
    }
}

int main()
{
    const int N = 10; // Number of vectors
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Allocate host memory for input and output vectors
    glm::vec2 h_a[N], h_b[N], h_result[N];

    // Initialize host vectors
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = glm::vec2(i, i);         // a = (i, i)
        h_b[i] = glm::vec2(i * 2, i * 3); // b = (2i, 3i)
    }

    // Allocate device memory
    glm::vec2 *d_a, *d_b, *d_result;
    CUDA_CHECK(cudaMalloc((void **)&d_a, N * sizeof(glm::vec2)));
    CUDA_CHECK(cudaMalloc((void **)&d_b, N * sizeof(glm::vec2)));
    CUDA_CHECK(cudaMalloc((void **)&d_result, N * sizeof(glm::vec2)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(glm::vec2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(glm::vec2), cudaMemcpyHostToDevice));

    // Launch the CUDA kernel
    addVectors<<<gridSize, blockSize>>>(d_a, d_b, d_result, N);

    // Copy the result back to the host
    CUDA_CHECK(cudaMemcpy(h_result, d_result, N * sizeof(glm::vec2), cudaMemcpyDeviceToHost));

    // Print the results
    std::cout << "Results:" << std::endl;
    for (int i = 0; i < N; ++i)
    {
        std::cout << "h_a[" << i << "] = (" << h_a[i].x << ", " << h_a[i].y << "), "
                  << "h_b[" << i << "] = (" << h_b[i].x << ", " << h_b[i].y << "), "
                  << "h_result[" << i << "] = (" << h_result[i].x << ", " << h_result[i].y << ")" << std::endl;
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_result));

    return 0;
}