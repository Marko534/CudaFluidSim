#include <iostream>
#include <cuda_runtime.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// CUDA kernel: computes a simple color gradient.
// Each thread computes one pixel's RGB values.
__global__ void computeGradient(unsigned char *img, int width, int height)
{
    // Compute the global pixel index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int index = (y * width + x) * 3; // 3 channels (RGB)

    // Compute gradient values (for example purposes)
    img[index + 0] = static_cast<unsigned char>((x / (float)width) * 255);  // Red channel
    img[index + 1] = static_cast<unsigned char>((y / (float)height) * 255); // Green channel
    img[index + 2] = 128;                                                   // Blue channel fixed (or computed by another formula)
}

int main()
{
    // Image dimensions
    const int width = 800;
    const int height = 600;
    const int channels = 3; // RGB

    // Allocate host memory for the image data
    size_t imageSize = width * height * channels * sizeof(unsigned char);
    unsigned char *h_img = new unsigned char[width * height * channels];

    // Allocate device memory
    unsigned char *d_img = nullptr;
    cudaMalloc((void **)&d_img, imageSize);

    // Define CUDA kernel launch configuration
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    // Launch CUDA kernel to compute the gradient image
    computeGradient<<<gridDim, blockDim>>>(d_img, width, height);
    cudaDeviceSynchronize();

    // Copy the result back to host memory
    cudaMemcpy(h_img, d_img, imageSize, cudaMemcpyDeviceToHost);

    // Save the image as a PNG using stb_image_write.
    // For PNG, the stride (number of bytes per row) is width * channels.
    stbi_write_png("output.png", width, height, channels, h_img, width * channels);

    // Print message and cleanup
    std::cout << "Image saved to output.png" << std::endl;
    cudaFree(d_img);
    delete[] h_img;

    return 0;
}
