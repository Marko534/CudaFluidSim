cmake_minimum_required(VERSION 3.8)
project(fluid_dynamics LANGUAGES CXX CUDA)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(CUDA REQUIRED)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# Suppress GLM warnings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress=20012")

# Add executable
add_executable(main main.cu)

# Link CUDA libraries properly
target_link_libraries(main PRIVATE
    ${CUDA_LIBRARIES}      # Use the correct CUDA runtime libraries
    cufft                  # Add cuFFT library
    OpenGL::GL            # Use modern CMake target for OpenGL
    glfw                  # Use the target provided by glfw3
    GLEW::GLEW            # Use modern CMake target for GLEW
    ${GLFW3_LIBRARIES}    # Explicitly link GLFW libraries (if needed)
    ${GLEW_LIBRARIES}     # Explicitly link GLEW libraries (if needed)
    ${OPENGL_LIBRARIES}   # Explicitly link OpenGL libraries (if needed)
    wayland-client        # Wayland client library
    xkbcommon             # XKB common library
    X11                   # X11 library
    Xi                    # X Input extension
    Xrandr                # X RandR extension
    Xcursor               # X Cursor library
    Xinerama              # Xinerama extension
    pthread               # POSIX threads
    dl                    # Dynamic linking library
    rt                    # Real-time library
)

# Include directories
target_include_directories(main PRIVATE
    ${CUDA_INCLUDE_DIRS}   # Include CUDA headers
    ${GLFW3_INCLUDE_DIRS}  # Include GLFW headers
    ${GLEW_INCLUDE_DIRS}   # Include GLEW headers
    ${OPENGL_INCLUDE_DIRS} # Include OpenGL headers
)

# Fix CUDA linking
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)