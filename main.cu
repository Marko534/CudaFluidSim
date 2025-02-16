#include <iostream>
#include <cuda_runtime.h>
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

int main()
{

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
