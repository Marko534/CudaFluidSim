cmake_minimum_required(VERSION 3.18)
project(fluid_dynamics LANGUAGES CXX CUDA)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(GLEW REQUIRED)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Add executable
add_executable(main main.cu)

# Set CUDA architectures
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES 75)

# Set CUDA flags using target_compile_options instead
target_compile_options(main PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20012>
)

# Add CUDA include directories
target_include_directories(main PRIVATE 
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(main PRIVATE
    CUDA::cudart
    CUDA::cufft
    OpenGL::GL
    glfw
    GLEW::GLEW
)

# Additional libraries for Linux
if(UNIX AND NOT APPLE)
    target_link_libraries(main PRIVATE
        X11
        pthread
        dl
    )
endif()

# Set CUDA properties
set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)