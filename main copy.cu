#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <cmath>

#define N 512
#define BLOCK_SIZE 16

// Simulation parameters
const float dt = 0.1f;
const float diffusion = 0.0001f;
const float viscosity = 0.0001f;

// CUDA kernels
__global__ void advect(float2 *v, float2 *v_prev, float *dens, float *dens_prev)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N)
        return;

    // Backtrack position with corrected clamping
    float2 v_prev_val = v_prev[i + N * j];
    float x = i - dt * N * v_prev_val.x;
    float y = j - dt * N * v_prev_val.y;

    x = fmaxf(0.5f, fminf(N - 1.5f, x));
    y = fmaxf(0.5f, fminf(N - 1.5f, y));

    int i0 = (int)x, i1 = i0 + 1;
    int j0 = (int)y, j1 = j0 + 1;

    float s1 = x - i0;
    float s0 = 1.0f - s1;
    float t1 = y - j0;
    float t0 = 1.0f - t1;

    // Velocity interpolation
    float2 v00 = v_prev[i0 + N * j0];
    float2 v01 = v_prev[i0 + N * j1];
    float2 v10 = v_prev[i1 + N * j0];
    float2 v11 = v_prev[i1 + N * j1];

    float vx = s0 * (t0 * v00.x + t1 * v01.x) + s1 * (t0 * v10.x + t1 * v11.x);
    float vy = s0 * (t0 * v00.y + t1 * v01.y) + s1 * (t0 * v10.y + t1 * v11.y);
    v[i + N * j] = make_float2(vx, vy);

    // Density interpolation
    dens[i + N * j] = s0 * (t0 * dens_prev[i0 + N * j0] + t1 * dens_prev[i0 + N * j1]) +
                      s1 * (t0 * dens_prev[i1 + N * j0] + t1 * dens_prev[i1 + N * j1]);
}

__global__ void jacobi(float *x, float *x0, float alpha, float beta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i <= 0 || i >= N - 1 || j <= 0 || j >= N - 1)
        return;

    float left = x0[(i - 1) + N * j];
    float right = x0[(i + 1) + N * j];
    float top = x0[i + N * (j - 1)];
    float bottom = x0[i + N * (j + 1)];

    x[i + N * j] = (left + right + top + bottom + alpha * x0[i + N * j]) / beta;
}

// Visualization functions
GLFWwindow *initGL()
{
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW\n";
        return nullptr;
    }
    GLFWwindow *window = glfwCreateWindow(N, N, "Fluid Simulation", NULL, NULL);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);
    return window;
}

void renderTexture(GLuint texID)
{
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texID);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(-1, -1);
    glTexCoord2f(1, 0);
    glVertex2f(1, -1);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(-1, 1);
    glEnd();
}

int main()
{
    // Initialize OpenGL (and create an OpenGL context)
    GLFWwindow *window = initGL();
    if (!window)
        return -1;

    // Create and configure the texture.
    // Using GL_R32F for a single-channel float texture (one float per pixel)
    GLuint texID;
    glGenTextures(1, &texID);
    glBindTexture(GL_TEXTURE_2D, texID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, N, N, 0, GL_RED, GL_FLOAT, NULL);

    // Allocate CUDA resources
    float2 *v, *v_prev;
    float *dens, *dens_prev;
    cudaMalloc(&v, N * N * sizeof(float2));
    cudaMalloc(&v_prev, N * N * sizeof(float2));
    cudaMalloc(&dens, N * N * sizeof(float));
    cudaMalloc(&dens_prev, N * N * sizeof(float));

    // Initialize device arrays (to avoid using uninitialized data)
    cudaMemset(v, 0, N * N * sizeof(float2));
    cudaMemset(v_prev, 0, N * N * sizeof(float2));
    cudaMemset(dens, 0, N * N * sizeof(float));
    cudaMemset(dens_prev, 0, N * N * sizeof(float));

    // Register the OpenGL texture with CUDA
    cudaGraphicsResource *cuda_tex;
    cudaGraphicsGLRegisterImage(&cuda_tex, texID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

    // Set up grid and block sizes for the kernels
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(N / BLOCK_SIZE, N / BLOCK_SIZE);

    while (!glfwWindowShouldClose(window))
    {
        // Run simulation kernel(s)
        advect<<<blocks, threads>>>(v, v_prev, dens, dens_prev);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA error (advect): " << cudaGetErrorString(err) << std::endl;
            break;
        }
        // (Additional kernels such as diffusion, projection, etc. would be called here)

        // Update the OpenGL texture with the density field
        cudaArray *tex_array;
        cudaGraphicsMapResources(1, &cuda_tex);
        cudaGraphicsSubResourceGetMappedArray(&tex_array, cuda_tex, 0, 0);
        cudaMemcpyToArray(tex_array, 0, 0, dens, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &cuda_tex);

        // Render the texture to the screen
        glClear(GL_COLOR_BUFFER_BIT);
        renderTexture(texID);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup resources
    cudaFree(v);
    cudaFree(v_prev);
    cudaFree(dens);
    cudaFree(dens_prev);
    cudaGraphicsUnregisterResource(cuda_tex);
    glfwTerminate();
    return 0;
}
