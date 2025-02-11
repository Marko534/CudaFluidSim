#include <GL/glew.h> // Must be included first
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cufft.h>

const int W = 512, H = 512;
const float dt = 0.1f, visc = 0.001f;

// CUDA-OpenGL interoperability
struct cudaGraphicsResource *cuda_vbo;
GLuint vbo;

// Fluid simulation resources
cufftHandle planR2C, planC2R;
cudaArray *velArray;
cudaTextureObject_t texVel; // Use cudaTextureObject_t instead of texture<>

__global__ void addForces(float2 *vel, int2 pos, float2 force, int r)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;

    int dx = x - pos.x;
    int dy = y - pos.y;
    float s = 1.f / (1.f + dx * dx * dx * dx + dy * dy * dy * dy);
    vel[y * W + x].x += s * force.x;
    vel[y * W + x].y += s * force.y;
}

__global__ void advectVel(float2 *vel, cudaTextureObject_t texOldVel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;

    float2 v;
    surf2Dread(&v, texOldVel, x * sizeof(float2), y);
    float2 pos = make_float2(x - dt * v.x, y - dt * v.y);

    float2 newVel;
    surf2Dread(&newVel, texOldVel, int(pos.x) * sizeof(float2), int(pos.y));
    vel[y * W + x] = newVel;
}

__global__ void updateParticles(float2 *particles, float2 *vel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= W * H)
        return;

    float2 p = particles[idx];
    int x = static_cast<int>(p.x * W);
    int y = static_cast<int>(p.y * H);
    x = max(0, min(x, W - 1));
    y = max(0, min(y, H - 1));

    p.x += vel[y * W + x].x * dt;
    p.y += vel[y * W + x].y * dt;

    // Boundary wrap
    p.x = fmod(p.x + 1.f, 1.f);
    p.y = fmod(p.y + 1.f, 1.f);

    particles[idx] = p;
}

void initSimulation()
{
    // Initialize CUDA-OpenGL interop
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, W * H * sizeof(float2), 0, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cuda_vbo, vbo, cudaGraphicsMapFlagsWriteDiscard);

    // Initialize velocity field
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();
    cudaMallocArray(&velArray, &desc, W, H);

    // Create CUFFT plans
    cufftPlan2d(&planR2C, H, W, CUFFT_R2C);
    cufftPlan2d(&planC2R, H, W, CUFFT_C2R);

    // Create and bind texture object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = velArray;

    cudaTextureDesc texDesc = {};
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(&texVel, &resDesc, &texDesc, nullptr);
}

void stepSimulation(int2 mousePos, float2 force)
{
    static float2 *d_vel;
    static cufftComplex *d_velC;

    // Add forces
    dim3 blocks(W / 16, H / 16);
    dim3 threads(16, 16);
    addForces<<<blocks, threads>>>(d_vel, mousePos, force, 20);

    // Advection
    advectVel<<<blocks, threads>>>(d_vel, texVel);

    // FFT-based diffusion and projection (simplified)
    cufftExecR2C(planR2C, (cufftReal *)d_vel, d_velC);
    // ... Diffusion and projection in frequency domain ...
    cufftExecC2R(planC2R, d_velC, (cufftReal *)d_vel);

    // Update particles
    float2 *d_part;
    cudaGraphicsMapResources(1, &cuda_vbo);
    size_t size;
    cudaGraphicsResourceGetMappedPointer((void **)&d_part, &size, cuda_vbo);
    updateParticles<<<(W * H + 255) / 256, 256>>>(d_part, d_vel);
    cudaGraphicsUnmapResources(1, &cuda_vbo);
}

void render()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(2, GL_FLOAT, 0, 0);
    glDrawArrays(GL_POINTS, 0, W * H);
    glDisableClientState(GL_VERTEX_ARRAY);
}

int main()
{
    glfwInit();
    GLFWwindow *window = glfwCreateWindow(W, H, "CUDA Fluid", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();

    initSimulation();

    while (!glfwWindowShouldClose(window))
    {
        // Get mouse input and calculate force
        // stepSimulation(mousePos, force);
        render();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaGraphicsUnregisterResource(cuda_vbo);
    glDeleteBuffers(1, &vbo);
    cufftDestroy(planR2C);
    cufftDestroy(planC2R);
    cudaDestroyTextureObject(texVel);
    cudaFreeArray(velArray);
    glfwTerminate();
    return 0;
}
