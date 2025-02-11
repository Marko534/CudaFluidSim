cmake_minimum_required(VERSION 3.8)
project(fluid_dynamics LANGUAGES CXX CUDA)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

add_executable(main main.cu)

target_link_libraries(main PRIVATE
    cudart
    OpenGL::GL
    glfw
    GLEW::GLEW
    wayland-client
    xkbcommon
    X11
    Xi
    Xrandr
    Xcursor
    Xinerama
    pthread
    dl
    rt
)