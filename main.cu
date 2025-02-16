#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>
#include <time.h>
#include <string.h>
#include <glm/glm.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

cudaError_t ercall;
#define CCALL(call)                                                                                                    \
    ercall = call;                                                                                                     \
    if (cudaSuccess != ercall)                                                                                         \
    {                                                                                                                  \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(ercall)); \
        exit(EXIT_FAILURE);                                                                                            \
    }

#define grid_l 480
#define grid_h 270

#define overrelax_const 1.0f

__device__ char *vectors;
__device__ char *vectorBuffer;

__device__ bool barrier[grid_l * grid_h];

// Const for numb of vectors
#define numHorizontal ((grid_l + 1) * grid_h)
#define numVertical ((grid_h + 1) * grid_l)

/*
 Cast the raw memory to glm::vec2 pointers
*/
#define horizontalVectors ((glm::vec2 *)vectors)
#define verticalVectors ((glm::vec2 *)(vectors + numHorizontal * sizeof(glm::vec2)))

#define horizontalVectorsBuffer ((glm::vec2 *)vectorBuffer)
#define verticalVectorsBuffer ((glm::vec2 *)(vectorBuffer + numHorizontal * sizeof(glm::vec2)))

#define horizontalVectorsCPU ((glm::vec2 *)cpuVecs)
#define verticalVectorsCPU ((glm::vec2 *)(cpuVecs + numHorizontal * sizeof(glm::vec2)))

// indexing macros remain the same
#define rightVecIndex(cellX, cellY) horizontalVecIndex(cellX + 1, cellY)
#define leftVecIndex(cellX, cellY) horizontalVecIndex(cellX, cellY)
#define upVecIndex(cellX, cellY) verticalVecIndex(cellX, cellY)
#define downVecIndex(cellX, cellY) verticalVecIndex(cellX, (cellY + 1))

#define verticalVecIndex(x, y) (x + y * (grid_l))
#define horizontalVecIndex(x, y) (x + y * (grid_l + 1))

#define inVerticalBounds(x, y) ((x) >= 0 && (x) < grid_l && (y) >= 0 && (y) <= grid_h)
#define inHorizontalBounds(x, y) ((x) >= 0 && (x) <= grid_l && (y) >= 0 && (y) < grid_h)
#define inCellBounds(x, y) ((x) >= 0 && (x) < grid_l && (y) >= 0 && (y) < grid_h)

#define cellXFromPos(p) (int)p.x
#define cellYFromPos(p) (int)p.y

inline __device__ void init_vec()
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int x = id % grid_l;
    const int y = id / grid_l;

    horizontalVectors[rightVecIndex(x, y)] = glm::vec2(0.0f, 0.0f);
    verticalVectors[upVecIndex(x, y)] = glm::vec2(0.0f, 0.0f);
    if (y == grid_h - 1)
    {
        verticalVectors[downVecIndex(x, y)] = glm::vec2(0.0f, 0.0f);
    }
    if (x == 0)
    {
        horizontalVectors[leftVecIndex(x, y)] = glm::vec2(0.0f, 0.0f);
    }
}

inline __device__ void init_vecBuffer()
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int x = id % grid_l;
    const int y = id / grid_l;

    horizontalVectorsBuffer[rightVecIndex(x, y)] = glm::vec2(0.0f, 0.0f);
    verticalVectorsBuffer[upVecIndex(x, y)] = glm::vec2(0.0f, 0.0f);
    if (y == grid_h - 1)
    {
        verticalVectorsBuffer[downVecIndex(x, y)] = glm::vec2(0.0f, 0.0f);
    }
    if (x == 0)
    {
        horizontalVectorsBuffer[leftVecIndex(x, y)] = glm::vec2(0.0f, 0.0f);
    }
}

inline __device__ void set_horizontal_vec_cell(const glm::vec2 v, const int x, const int y)
{
    horizontalVectors[rightVecIndex(x, y)] = v;
    horizontalVectors[leftVecIndex(x, y)] = v;
}

inline __device__ void set_vertical_vec_cell(const glm::vec2 v, const int x, const int y)
{
    verticalVectors[upVecIndex(x, y)] = v;
    verticalVectors[downVecIndex(x, y)] = v;
}

__global__ void setHorizontalVec(const glm::vec2 v, const int x, const int y)
{
    set_horizontal_vec_cell(v, x, y);
}

__global__ void setVerticalVec(const glm::vec2 v, const int x, const int y)
{
    set_vertical_vec_cell(v, x, y);
}

void setHorVecs(const glm::vec2 v, const int x, const int y)
{
    setHorizontalVec<<<1, 1>>>(v, x, y);
}

void setVertVecs(const glm::vec2 v, const int x, const int y)
{
    setVerticalVec<<<1, 1>>>(v, x, y);
}

// init barrier
inline __device__ void init_barrier()
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    barrier[id] = false;
}

// set a barrier for a single cell
inline __device__ void set_barrier(const int x, const int y)
{
    barrier[x + y * grid_l] = true;
}

__global__ void setBarrier(const int x, const int y)
{
    set_barrier(x, y);
}

void setBar(const int x, const int y)
{
    setBarrier<<<1, 1>>>(x, y);
}

// reset kernels
__global__ void resetVectors()
{
    init_vec();
}

__global__ void resetVectorsBuffer()
{
    init_vecBuffer();
}

__global__ void resetBarriers()
{
    init_barrier();
}

void resetVecs()
{
    resetVectors<<<512, grid_l * grid_h / 512>>>();
}

void resetVecsBuf()
{
    resetVectorsBuffer<<<512, grid_l * grid_h / 512>>>();
}

void resetBars()
{
    resetBarriers<<<512, grid_l * grid_h / 512>>>();
}

// Calculate divergence for a single cell using the current vector field.
inline __device__ double calc_divergence(const int x, const int y)
{
    return (verticalVectors[upVecIndex(x, y)].y * inCellBounds(x, y) -
            verticalVectors[downVecIndex(x, y)].y * inCellBounds(x, y) +
            horizontalVectors[rightVecIndex(x, y)].x * inCellBounds(x, y) -
            horizontalVectors[leftVecIndex(x, y)].x * inCellBounds(x, y)) *
           overrelax_const;
}

#define L 0
#define R 1
#define T 2
#define B 3
inline __device__ void apply_divergence(const int x, const int y)
{
    bool affected_cells[4];
    ((char32_t *)affected_cells)[0] = 0;
    unsigned char num_affected = 0;

    if (barrier[x + y * grid_l])
    {
        return;
    }

    // Check left/right neighbors
#pragma unroll
    for (int xo = -1; xo <= 1; xo += 2)
    {
        if (!inCellBounds(x + xo, y) || barrier[x + xo + y * grid_l])
        {
            continue;
        }
        num_affected += 1;
        affected_cells[(xo + 1) / 2] = true;
    }

    // Check top/bottom neighbors
#pragma unroll
    for (int yo = -1; yo <= 1; yo += 2)
    {
        if (!inCellBounds(x, y + yo) || barrier[x + (y + yo) * grid_l])
        {
            continue;
        }
        num_affected += 1;
        affected_cells[(yo + 5) / 2] = true;
    }

    if (num_affected == 0)
    {
        return;
    }

    const float divergence = calc_divergence(x, y) / (float)num_affected;

    // Distribute the divergence correction to each neighbor that is not blocked.
    verticalVectorsBuffer[upVecIndex(x, y)].y -= divergence * affected_cells[T];      // up
    verticalVectorsBuffer[downVecIndex(x, y)].y += divergence * affected_cells[B];    // down
    horizontalVectorsBuffer[rightVecIndex(x, y)].x -= divergence * affected_cells[R]; // right
    horizontalVectorsBuffer[leftVecIndex(x, y)].x += divergence * affected_cells[L];  // left
}

// Divergence equations are solved using Gaussian elimination in two passes (checkerboard “white” and “black” cells)
#define threads_divergence 256
#define blocks_divergence ((grid_l * grid_h) + threads_divergence - 1) / threads_divergence

// divergence kernel for "white" cells
__global__ void divergenceGaussianW()
{
    const int cellId = threadIdx.x + blockIdx.x * blockDim.x;
    const int cellY = cellId * 2 / grid_l;
    const int cellX = 2 * cellId % grid_l + 1 * (cellY % 2 == 0);

    if (cellId >= grid_l * grid_h)
    {
        return;
    }
    apply_divergence(cellX, cellY);
}

// divergence kernel for "black" cells
__global__ void divergenceGaussianB()
{
    const int cellId = threadIdx.x + blockIdx.x * blockDim.x;
    const int cellY = cellId * 2 / grid_l;
    const int cellX = 2 * cellId % grid_l + 1 * (cellY % 2 == 1);

    if (cellId >= grid_l * grid_h)
    {
        return;
    }
    apply_divergence(cellX, cellY);
}

// add the buffer to the main array
__global__ void addBufferW()
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int cellId = id;
    if (cellId >= grid_l * grid_h)
        return; // avoid out-of-bounds

    verticalVectors[upVecIndex(cellId % grid_l, cellId / grid_l)].y += verticalVectorsBuffer[upVecIndex(cellId % grid_l, cellId / grid_l)].y;
    horizontalVectors[leftVecIndex(cellId % grid_l, cellId / grid_l)].x += horizontalVectorsBuffer[leftVecIndex(cellId % grid_l, cellId / grid_l)].x;

    if (cellId % grid_l == grid_l - 1)
    {
        horizontalVectors[rightVecIndex(cellId % grid_l, cellId / grid_l)].x += horizontalVectorsBuffer[rightVecIndex(cellId % grid_l, cellId / grid_l)].x;
    }
    if (cellId / grid_l == grid_h - 1)
    {
        verticalVectors[downVecIndex(cellId % grid_l, cellId / grid_l)].y += verticalVectorsBuffer[downVecIndex(cellId % grid_l, cellId / grid_l)].y;
    }
}

void addBuf()
{
    int totalThreads = grid_l * grid_h;
    int blocks = (totalThreads + 511) / 512;
    addBufferW<<<blocks, 512>>>();
    CCALL(cudaDeviceSynchronize());
}

// CPU function to call the divergence solver kernels
void gaussianDivergenceSolver(const int passes)
{
    cudaDeviceSynchronize();
    for (int p = 0; p < passes; p++)
    {
        resetVecsBuf();
        divergenceGaussianB<<<threads_divergence, blocks_divergence>>>();
        cudaDeviceSynchronize();
        divergenceGaussianW<<<threads_divergence, blocks_divergence>>>();
        cudaDeviceSynchronize();
        addBuf();

        // call in reverse order
        // resetVecsBuf();
        // divergenceGaussianW<<<threads_divergence, blocks_divergence>>>();
        // cudaDeviceSynchronize();
        // divergenceGaussianB<<<threads_divergence, blocks_divergence>>>();
        // cudaDeviceSynchronize();
        // addBuf();
    }
}

//*****************************************************************************************************************************************************************************************
// Advection functions and kernel

#define threads_advection 512
#define blocks_advection grid_l *grid_h / threads_advection / 2

// advection kernel for "white" cells
__global__ void advectionKernelW()
{
    const int cellId = threadIdx.x + blockIdx.x * blockDim.x;
    const int cellY = cellId * 2 / grid_l;
    const int cellX = 2 * cellId % grid_l + 1 * (cellY % 2 == 0);

    if (!inCellBounds(cellX, cellY))
    {
        return;
    }

    // Use semi-Lagrangian backtracing. Note: we explicitly construct a glm::vec2.
    const glm::vec2 prev_pos = glm::vec2(cellX + 0.5f, cellY + 0.5f) - horizontalVectors[rightVecIndex(cellX, cellY)] - horizontalVectors[leftVecIndex(cellX, cellY)] - verticalVectors[upVecIndex(cellX, cellY)] - verticalVectors[downVecIndex(cellX, cellY)];

    const int prevCellX = (int)prev_pos.x;
    const int prevCellY = (int)prev_pos.y;

    const float xOffsetFromLeft = prev_pos.x - prevCellX;
    const float yOffsetFromTop = prev_pos.y - prevCellY;

    if (barrier[prevCellX + prevCellY * grid_l] || !inCellBounds(prevCellX, prevCellY) || !inCellBounds(cellX, cellY))
    {
        return;
    }

    horizontalVectorsBuffer[rightVecIndex(cellX, cellY)] =
        horizontalVectors[rightVecIndex(prevCellX, prevCellY)] * xOffsetFromLeft * (cellX == grid_l - 1 ? 1.0f : 0.5f);
    horizontalVectorsBuffer[leftVecIndex(cellX, cellY)] =
        horizontalVectors[leftVecIndex(prevCellX, prevCellY)] * (1 - xOffsetFromLeft) * (cellX == 0 ? 1.0f : 0.5f);

    verticalVectorsBuffer[upVecIndex(cellX, cellY)] =
        verticalVectors[upVecIndex(prevCellX, prevCellY)] * (1 - yOffsetFromTop) * (cellY == 0 ? 1.0f : 0.5f);
    verticalVectorsBuffer[downVecIndex(cellX, cellY)] =
        verticalVectors[downVecIndex(prevCellX, prevCellY)] * yOffsetFromTop * (cellY == grid_h - 1 ? 1.0f : 0.5f);
}

__global__ void advectionKernelB()
{
    const int cellId = threadIdx.x + blockIdx.x * blockDim.x;
    const int cellY = cellId * 2 / grid_l;
    const int cellX = 2 * cellId % grid_l + 1 * (cellY % 2 == 1);

    if (!inCellBounds(cellX, cellY))
    {
        return;
    }

    const glm::vec2 prev_pos = glm::vec2(cellX + 0.5f, cellY + 0.5f) - horizontalVectors[rightVecIndex(cellX, cellY)] - horizontalVectors[leftVecIndex(cellX, cellY)] - verticalVectors[upVecIndex(cellX, cellY)] - verticalVectors[downVecIndex(cellX, cellY)];

    const int prevCellX = (int)prev_pos.x;
    const int prevCellY = (int)prev_pos.y;

    const float xOffsetFromLeft = prev_pos.x - prevCellX;
    const float yOffsetFromTop = prev_pos.y - prevCellY;

    if (barrier[prevCellX + prevCellY * grid_l] || !inCellBounds(prevCellX, prevCellY) || !inCellBounds(cellX, cellY))
    {
        return;
    }

    horizontalVectorsBuffer[rightVecIndex(cellX, cellY)] =
        horizontalVectorsBuffer[rightVecIndex(cellX, cellY)] +
        horizontalVectors[rightVecIndex(prevCellX, prevCellY)] * xOffsetFromLeft * (cellX == grid_l - 1 ? 1.0f : 0.5f);
    horizontalVectorsBuffer[leftVecIndex(cellX, cellY)] =
        horizontalVectorsBuffer[leftVecIndex(cellX, cellY)] +
        horizontalVectors[leftVecIndex(prevCellX, prevCellY)] * (1 - xOffsetFromLeft) * (cellX == 0 ? 1.0f : 0.5f);

    verticalVectorsBuffer[upVecIndex(cellX, cellY)] =
        verticalVectorsBuffer[upVecIndex(cellX, cellY)] +
        verticalVectors[upVecIndex(prevCellX, prevCellY)] * (1 - yOffsetFromTop) * (cellY == 0 ? 1.0f : 0.5f);
    verticalVectorsBuffer[downVecIndex(cellX, cellY)] =
        verticalVectorsBuffer[downVecIndex(cellX, cellY)] +
        verticalVectors[downVecIndex(prevCellX, prevCellY)] * yOffsetFromTop * (cellY == grid_h - 1 ? 1.0f : 0.5f);
}

__global__ void copyFromBuffer()
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int x = id % grid_l;
    const int y = id / grid_l;

    horizontalVectors[rightVecIndex(x, y)] = horizontalVectors[rightVecIndex(x, y)];
    verticalVectors[downVecIndex(x, y)] = verticalVectorsBuffer[downVecIndex(x, y)];

    if (x == 0)
    {
        horizontalVectors[leftVecIndex(x, y)] = horizontalVectors[leftVecIndex(x, y)];
    }
    if (y == 0)
    {
        verticalVectors[upVecIndex(x, y)] = verticalVectorsBuffer[upVecIndex(x, y)];
    }
}

void semiLagrangianAdvection()
{
    CCALL(cudaDeviceSynchronize());
    advectionKernelW<<<threads_advection, blocks_advection>>>();
    CCALL(cudaDeviceSynchronize());
    advectionKernelB<<<threads_advection, blocks_advection>>>();
    CCALL(cudaDeviceSynchronize());
    copyFromBuffer<<<threads_advection, blocks_advection * 2>>>();
    CCALL(cudaDeviceSynchronize());
}

//*****************************************************************************************************************************************************************************************
// Allocation and memory moving functions

// Notice we change sizeof(vec2) to sizeof(glm::vec2)
char cpuVecs[(grid_l * (grid_h + 1) + grid_h * (grid_l + 1)) * sizeof(glm::vec2)];
bool cpuBarrier[grid_l * grid_h];

char *deviceVecPointer;
char *deviceVecBufferPointer;

void allocDeviceVars()
{
#ifdef DEBUG
    cudaError_t m1, m2, c1, c2;
    m1 = cudaMalloc((void **)(&deviceVecPointer), (grid_l * (grid_h + 1) + grid_h * (grid_l + 1)) * sizeof(glm::vec2));
    c1 = cudaMemcpyToSymbol(vectors, &deviceVecPointer, sizeof(char *));
    m2 = cudaMalloc((void **)(&deviceVecBufferPointer), (grid_l * (grid_h + 1) + grid_h * (grid_l + 1)) * sizeof(glm::vec2));
    c2 = cudaMemcpyToSymbol(vectorBuffer, &deviceVecBufferPointer, sizeof(char *));
    printf("alloc one     malloc: %s | copy: %s\n", cudaGetErrorString(m1), cudaGetErrorString(c1));
    printf("alloc two     malloc: %s | copy: %s\n", cudaGetErrorString(m2), cudaGetErrorString(c2));
#else
    cudaMalloc((void **)(&deviceVecPointer), (grid_l * (grid_h + 1) + grid_h * (grid_l + 1)) * sizeof(glm::vec2));
    cudaMemcpyToSymbol(vectors, &deviceVecPointer, sizeof(char *));
    cudaMalloc((void **)(&deviceVecBufferPointer), (grid_l * (grid_h + 1) + grid_h * (grid_l + 1)) * sizeof(glm::vec2));
    cudaMemcpyToSymbol(vectorBuffer, &deviceVecBufferPointer, sizeof(char *));
#endif
}

void moveMainArrayToCPU()
{
    cudaError_t e = cudaMemcpy(cpuVecs, deviceVecPointer, (grid_l * (grid_h + 1) + grid_h * (grid_l + 1)) * sizeof(glm::vec2), cudaMemcpyDeviceToHost);
#ifdef DEBUG
    printf("copy vecs: %s\n", cudaGetErrorString(e));
#endif
}

void moveBarrierToCPU()
{
    cudaError_t e = cudaMemcpyFromSymbol(cpuBarrier, barrier, grid_l * grid_h * sizeof(bool));
#ifdef DEBUG
    printf("copy barrier: %s\n", cudaGetErrorString(e));
#endif
}

/*
    Visualization functions
*/
struct color
{
    unsigned char r, g, b;
    __host__ __device__ color() : r(0), g(0), b(0) {}
    __host__ __device__ color(float red, float green, float blue) : r(red), g(green), b(blue) {}
};

color sampleFieldVelocityDirectionalMagnitude(const int x, const int y, float threshold)
{
    const float totalPos = fabs(horizontalVectorsCPU[rightVecIndex(x, y)].x) + fabs(verticalVectorsCPU[upVecIndex(x, y)].y);
    const float totalNeg = fabs(horizontalVectorsCPU[leftVecIndex(x, y)].x) + fabs(verticalVectorsCPU[downVecIndex(x, y)].y);

    float magnitudePos = totalPos / threshold;
    float magnitudeNeg = totalNeg / threshold;

    magnitudePos = (magnitudePos > 1.0f) ? 1.0f : magnitudePos;
    magnitudeNeg = (magnitudeNeg > 1.0f) ? 1.0f : magnitudeNeg;

    return color(magnitudePos * 255, 0, magnitudeNeg * 255);
}

unsigned char cpuColors[grid_l * grid_h * 3];

void fillColorArray(float threshold)
{

    for (int x = 0; x < grid_l; x++)
    {
        for (int y = grid_h - 1; y >= 0; y--)
        {
            ((color *)cpuColors)[x + (grid_h - y - 1) * grid_l] = sampleFieldVelocityDirectionalMagnitude(x, y, threshold);
            if (cpuBarrier[x + y * grid_l])
            {
                ((color *)cpuColors)[x + (grid_h - 1 - y) * grid_l] = color(255, 255, 255);
            }
        }
    }
}

void updateFluid(float v)
{
    gaussianDivergenceSolver(512);
    semiLagrangianAdvection();
}

int main()
{
    allocDeviceVars();
    resetVecs();
    resetBars();

    float fluidvel = 1.0f;

    // Circle barrier at (150, 150)
    int xcenter = 150, ycenter = 150;
    float radius = 30.0f;
    for (int x = 0; x < grid_l; x++)
    {
        for (int y = 0; y < grid_h; y++)
        {
            if (sqrtf((x - xcenter) * (x - xcenter) + (y - ycenter) * (y - ycenter)) < radius)
            {
                setBar(x, y);
            }
        }
    }

    // Square barrier at a different center, for example (300, 150)
    xcenter = 300;
    ycenter = 150;
    int side_length = 60;
    int half_side = side_length / 2;
    for (int x = 0; x < grid_l; x++)
    {
        for (int y = 0; y < grid_h; y++)
        {
            if (x >= xcenter - half_side && x < xcenter + half_side &&
                y >= ycenter - half_side && y < ycenter + half_side)
            {
                setBar(x, y);
            }
        }
    }
    /*
        Initilining directional vecs
    */
    // Initialize horizontal vectors (e.g., on the left boundary) for x-direction flow
    for (int y = 0; y < grid_h; y++)
    {
        setHorVecs(glm::vec2(fluidvel, 0.0f), 0, y);
    }

    moveBarrierToCPU();

    const unsigned int totalIterations = 2024;
    for (unsigned int iter = 1; iter <= totalIterations; iter++)
    {
        updateFluid(fluidvel);

        if ((iter & (iter - 1)) == 0 && iter >= 16)
        {
            moveMainArrayToCPU();
            fillColorArray(fluidvel);

            char filename[64];
            sprintf(filename, "output_%u.png", iter);
            stbi_write_png(filename, grid_l, grid_h, 3, cpuColors, grid_l * 3);
            printf("Saved image at iteration %u: %s\n", iter, filename);
        }
    }

    return 0;
}
