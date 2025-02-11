#include <GL/glew.h> // Must be first!
#include <GL/glut.h>
#include <GL/gl.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector_types.h> // Ensures we use CUDA's built-in float2

#define WIDTH 512
#define HEIGHT 512
#define NUM_PARTICLES 10000

GLuint vbo;                              // OpenGL vertex buffer object
cudaGraphicsResource *cuda_vbo_resource; // CUDA Graphics Resource (to map the VBO)

// CUDA kernel that updates particle positions.
// For demonstration, this kernel rotates each particle around the center.
__global__ void updateParticles(float2 *particles, int num_particles, float dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles)
        return;

    // Simple rotation around the center (0.5, 0.5)
    float angle = dt * 0.1f; // You can change this for different speeds
    float cx = 0.5f, cy = 0.5f;
    float x = particles[idx].x;
    float y = particles[idx].y;
    float dx = x - cx;
    float dy = y - cy;

    // Rotation formula
    float cosA = cosf(angle);
    float sinA = sinf(angle);
    float newX = cosA * dx - sinA * dy + cx;
    float newY = sinA * dx + cosA * dy + cy;

    particles[idx].x = newX;
    particles[idx].y = newY;
}

// Initialize the VBO and register it with CUDA.
void initVBO()
{
    // Generate and bind the VBO
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // Allocate space for NUM_PARTICLES 2D positions
    glBufferData(GL_ARRAY_BUFFER, NUM_PARTICLES * sizeof(float2), nullptr, GL_DYNAMIC_DRAW);

    // Map the buffer to initialize the particle positions
    float2 *ptr = (float2 *)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    if (!ptr)
    {
        std::cerr << "Could not map VBO to initialize data." << std::endl;
        exit(EXIT_FAILURE);
    }
    // Initialize particles at random positions (in normalized [0,1] coordinates)
    for (int i = 0; i < NUM_PARTICLES; i++)
    {
        ptr[i].x = static_cast<float>(rand()) / RAND_MAX;
        ptr[i].y = static_cast<float>(rand()) / RAND_MAX;
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Register the VBO with CUDA for writing
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaGraphicsGLRegisterBuffer failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// OpenGL display callback: clear the screen and draw the particles.
void display()
{
    glClear(GL_COLOR_BUFFER_BIT);

    // Enable and bind the VBO as a vertex array
    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(2, GL_FLOAT, 0, 0);

    // Draw particles as points
    glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
    glutSwapBuffers();
}

// Idle update callback: update particle positions using the CUDA kernel.
void update()
{
    // Map the CUDA graphics resource (the VBO)
    float2 *dptr = nullptr;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cuda_vbo_resource);

    // Launch CUDA kernel to update particle positions.
    int threadsPerBlock = 256;
    int blocks = (NUM_PARTICLES + threadsPerBlock - 1) / threadsPerBlock;
    updateParticles<<<blocks, threadsPerBlock>>>(dptr, NUM_PARTICLES, 0.016f); // dt ~ 16ms/frame

    // Unmap the resource so OpenGL can use it again.
    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);

    // Post a redisplay event so that display() gets called.
    glutPostRedisplay();
}

int main(int argc, char **argv)
{
    // Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("CUDA OpenGL Dynamic Particle Simulation");

    // Initialize GLEW (required for OpenGL extensions)
    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
        std::cerr << "GLEW initialization failed: " << glewGetErrorString(err) << std::endl;
        return -1;
    }

    // Set clear color to black
    glClearColor(0.0, 0.0, 0.0, 1.0);

    // Initialize the VBO and register it with CUDA.
    initVBO();

    // Set up GLUT callbacks
    glutDisplayFunc(display);
    glutIdleFunc(update);

    // Start the main loop
    glutMainLoop();

    return 0;
}
