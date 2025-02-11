// fluid_simulation.cu
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <cmath>

#define N 512
#define BLOCK_SIZE 16

// Simulation parameters
const float dt = 0.1f;      // time step
const float diff = 0.0001f; // density diffusion coefficient
const float visc = 0.0001f; // viscosity (velocity diffusion)

// ---------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------

// Kernel: Add external force and inject density in a circular region.
__global__ void addForce(float2 *v, float *dens, float2 force, int centerX, int centerY, float radius)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N)
        return;
    int index = i + j * N;
    float dx = i - centerX;
    float dy = j - centerY;
    float dist = sqrtf(dx * dx + dy * dy);
    if (dist < radius)
    {
        float factor = (radius - dist) / radius;
        v[index].x += force.x * factor;
        v[index].y += force.y * factor;
        dens[index] += 100.0f * factor; // density injection (scale as needed)
    }
}

// Kernel: Diffuse the velocity field (Jacobi-style update).
__global__ void diffuseVelocity(float2 *v, float2 *v0, float visc, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i <= 0 || i >= N - 1 || j <= 0 || j >= N - 1)
        return;
    int index = i + j * N;
    float a = visc * dt;
    float2 left = v0[index - 1];
    float2 right = v0[index + 1];
    float2 top = v0[index - N];
    float2 bottom = v0[index + N];
    v[index].x = (v0[index].x + a * (left.x + right.x + top.x + bottom.x)) / (1.0f + 4.0f * a);
    v[index].y = (v0[index].y + a * (left.y + right.y + top.y + bottom.y)) / (1.0f + 4.0f * a);
}

// Kernel: Compute the divergence of the velocity field (using central differences).
__global__ void computeDivergence(float2 *v, float *div)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i <= 0 || i >= N - 1 || j <= 0 || j >= N - 1)
        return;
    int index = i + j * N;
    // Assuming grid spacing h = 1.0 (for simplicity)
    float v_right = v[index + 1].x;
    float v_left = v[index - 1].x;
    float v_top = v[index - N].y;
    float v_bot = v[index + N].y;
    div[index] = (v_right - v_left + v_bot - v_top) * 0.5f;
}

// Kernel: Solve for pressure using Jacobi iterations.
__global__ void pressureJacobi(float *p, float *p0, float *div)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i <= 0 || i >= N - 1 || j <= 0 || j >= N - 1)
        return;
    int index = i + j * N;
    p[index] = (div[index] + p0[index - 1] + p0[index + 1] + p0[index - N] + p0[index + N]) * 0.25f;
}

// Kernel: Subtract the gradient of pressure from the velocity field.
__global__ void projectVelocity(float2 *v, float *p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i <= 0 || i >= N - 1 || j <= 0 || j >= N - 1)
        return;
    int index = i + j * N;
    v[index].x -= 0.5f * (p[index + 1] - p[index - 1]);
    v[index].y -= 0.5f * (p[index + N] - p[index - N]);
}

// Kernel: Advect (move) the velocity field along itself.
__global__ void advectVelocity(float2 *v, float2 *v0, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N)
        return;
    int index = i + j * N;
    // Backtrace the particle position
    float2 pos = make_float2((float)i, (float)j);
    float2 v_val = v0[index];
    float dt0 = dt * N;
    float2 posBack = make_float2(pos.x - dt0 * v_val.x, pos.y - dt0 * v_val.y);
    posBack.x = fmaxf(0.5f, fminf(N - 1.5f, posBack.x));
    posBack.y = fmaxf(0.5f, fminf(N - 1.5f, posBack.y));
    int i0 = (int)posBack.x, i1 = i0 + 1;
    int j0 = (int)posBack.y, j1 = j0 + 1;
    float s1 = posBack.x - i0, s0 = 1.0f - s1;
    float t1 = posBack.y - j0, t0 = 1.0f - t1;
    float2 v00 = v0[i0 + j0 * N];
    float2 v01 = v0[i0 + j1 * N];
    float2 v10 = v0[i1 + j0 * N];
    float2 v11 = v0[i1 + j1 * N];
    v[index].x = s0 * (t0 * v00.x + t1 * v01.x) + s1 * (t0 * v10.x + t1 * v11.x);
    v[index].y = s0 * (t0 * v00.y + t1 * v01.y) + s1 * (t0 * v10.y + t1 * v11.y);
}

// Kernel: Advect the density field using the velocity field.
__global__ void advectDensity(float *dens, float *dens0, float2 *v, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N)
        return;
    int index = i + j * N;
    float2 pos = make_float2((float)i, (float)j);
    float2 v_val = v[index];
    float dt0 = dt * N;
    float2 posBack = make_float2(pos.x - dt0 * v_val.x, pos.y - dt0 * v_val.y);
    posBack.x = fmaxf(0.5f, fminf(N - 1.5f, posBack.x));
    posBack.y = fmaxf(0.5f, fminf(N - 1.5f, posBack.y));
    int i0 = (int)posBack.x, i1 = i0 + 1;
    int j0 = (int)posBack.y, j1 = j0 + 1;
    float s1 = posBack.x - i0, s0 = 1.0f - s1;
    float t1 = posBack.y - j0, t0 = 1.0f - t1;
    dens[index] = s0 * (t0 * dens0[i0 + j0 * N] + t1 * dens0[i0 + j1 * N]) +
                  s1 * (t0 * dens0[i1 + j0 * N] + t1 * dens0[i1 + j1 * N]);
}

// Kernel: Diffuse the density field.
__global__ void diffuseDensity(float *dens, float *dens0, float diff, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i <= 0 || i >= N - 1 || j <= 0 || j >= N - 1)
        return;
    int index = i + j * N;
    float a = diff * dt;
    float left = dens0[index - 1];
    float right = dens0[index + 1];
    float top = dens0[index - N];
    float bottom = dens0[index + N];
    dens[index] = (dens0[index] + a * (left + right + top + bottom)) / (1.0f + 4.0f * a);
}

// ---------------------------------------------------------------------
// OpenGL utility functions
// ---------------------------------------------------------------------

// Initialize GLFW and create an OpenGL context.
GLFWwindow *initGL()
{
    // Request an OpenGL 2.1 context (compatibility profile)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW\n";
        return nullptr;
    }
    GLFWwindow *window = glfwCreateWindow(N, N, "CUDA Fluid Simulation", NULL, NULL);
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
    // Initialize OpenGL
    GLFWwindow *window = initGL();
    if (!window)
        return -1;

    // *** CUDA–OpenGL Interop Initialization ***
    // Ensure CUDA uses the same device as OpenGL.
    cudaError_t err = cudaGLSetGLDevice(0);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaGLSetGLDevice failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Create and configure the texture.
    GLuint texID;
    glGenTextures(1, &texID);
    glBindTexture(GL_TEXTURE_2D, texID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, N, N, 0, GL_RED, GL_FLOAT, NULL);

    // Register the OpenGL texture with CUDA.
    cudaGraphicsResource *cuda_tex;
    err = cudaGraphicsGLRegisterImage(&cuda_tex, texID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaGraphicsGLRegisterImage failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Allocate CUDA device memory.
    size_t size_v = N * N * sizeof(float2);
    size_t size_d = N * N * sizeof(float);
    size_t size_p = N * N * sizeof(float);
    float2 *v, *v0;
    float *dens, *dens0;
    float *p, *p0, *div;
    cudaMalloc(&v, size_v);
    cudaMalloc(&v0, size_v);
    cudaMalloc(&dens, size_d);
    cudaMalloc(&dens0, size_d);
    cudaMalloc(&p, size_p);
    cudaMalloc(&p0, size_p);
    cudaMalloc(&div, size_p);

    // Initialize fields to zero.
    cudaMemset(v, 0, size_v);
    cudaMemset(v0, 0, size_v);
    cudaMemset(dens, 0, size_d);
    cudaMemset(dens0, 0, size_d);
    cudaMemset(p, 0, size_p);
    cudaMemset(p0, 0, size_p);
    cudaMemset(div, 0, size_p);

    // Set up CUDA grid.
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Main simulation loop.
    while (!glfwWindowShouldClose(window))
    {
        // 1. Add external force and density.
        int centerX = N / 2;
        int centerY = N / 2;
        float radius = 10.0f;
        float2 force = make_float2(0.0f, -1.0f);
        addForce<<<blocks, threads>>>(v, dens, force, centerX, centerY, radius);
        cudaDeviceSynchronize(); // Check for errors in kernel launch.

        // 2. Diffuse velocity.
        cudaMemcpy(v0, v, size_v, cudaMemcpyDeviceToDevice);
        diffuseVelocity<<<blocks, threads>>>(v, v0, visc, dt);
        cudaDeviceSynchronize();

        // 3. Compute divergence.
        computeDivergence<<<blocks, threads>>>(v, div);
        cudaDeviceSynchronize();

        // 4. Solve for pressure (20 Jacobi iterations).
        cudaMemset(p, 0, size_p);
        for (int k = 0; k < 20; k++)
        {
            cudaMemcpy(p0, p, size_p, cudaMemcpyDeviceToDevice);
            pressureJacobi<<<blocks, threads>>>(p, p0, div);
            cudaDeviceSynchronize();
        }

        // 5. Project velocity.
        projectVelocity<<<blocks, threads>>>(v, p);
        cudaDeviceSynchronize();

        // 6. Advect velocity.
        cudaMemcpy(v0, v, size_v, cudaMemcpyDeviceToDevice);
        advectVelocity<<<blocks, threads>>>(v, v0, dt);
        cudaDeviceSynchronize();

        // 7. Advect density.
        cudaMemcpy(dens0, dens, size_d, cudaMemcpyDeviceToDevice);
        advectDensity<<<blocks, threads>>>(dens, dens0, v, dt);
        cudaDeviceSynchronize();

        // 8. Diffuse density.
        cudaMemcpy(dens0, dens, size_d, cudaMemcpyDeviceToDevice);
        diffuseDensity<<<blocks, threads>>>(dens, dens0, diff, dt);
        cudaDeviceSynchronize();

        // Update the OpenGL texture with the density field.
        cudaArray *tex_array;
        err = cudaGraphicsMapResources(1, &cuda_tex, 0);
        if (err != cudaSuccess)
        {
            std::cerr << "cudaGraphicsMapResources failed: " << cudaGetErrorString(err) << std::endl;
            break;
        }
        err = cudaGraphicsSubResourceGetMappedArray(&tex_array, cuda_tex, 0, 0);
        if (err != cudaSuccess)
        {
            std::cerr << "cudaGraphicsSubResourceGetMappedArray failed: " << cudaGetErrorString(err) << std::endl;
            break;
        }
        cudaMemcpyToArray(tex_array, 0, 0, dens, size_d, cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &cuda_tex, 0);

        // Render the texture.
        glClear(GL_COLOR_BUFFER_BIT);
        renderTexture(texID);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup.
    cudaFree(v);
    cudaFree(v0);
    cudaFree(dens);
    cudaFree(dens0);
    cudaFree(p);
    cudaFree(p0);
    cudaFree(div);
    cudaGraphicsUnregisterResource(cuda_tex);
    glfwTerminate();
    return 0;
}