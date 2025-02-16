# CUDA Fluid Simulation

This project simulates fluid dynamics using CUDA for parallel computation. The simulation includes barriers (e.g., circles and squares) that affect the fluid flow, and the results are visualized as images saved at specific iterations.

## Requirements
- **CMake**
- **CUDA Toolkit**: Ensure you have the CUDA toolkit installed on your system.
- **GLM Library**: The OpenGL Mathematics (GLM) library is used for vector and matrix operations.
- **STB Image Library**: The `stb_image_write.h` library is used for saving the output images.

## Building the Project

1. **Install Dependencies**:
   - Install the CUDA toolkit from [NVIDIA's official website](https://developer.nvidia.com/cuda-toolkit).
   - Install the GLM library. You can typically install it via your package manager:
   For Fedora and RedHat based systems 
   ```bash
   sudo dnf install glm-devil
   ```

2. **Compile the Code**:
   - Use CMake

## Code Overview

- **Initialization**:
  - The grid size is defined as `720x480`.
  - Barriers (e.g., a circle and a square) are initialized to affect the fluid flow.
  - Horizontal vectors are initialized to create a flow in the x-direction.

- **Simulation Loop**:
  - The simulation runs for a total of 2024 iterations.
  - At each iteration, the fluid dynamics are updated using the `updateFluid` function, which solves the divergence and performs semi-Lagrangian advection.
  - Images are saved at iterations that are powers of 2.

- **Visualization**:
  - The fluid velocity is visualized using colors, where the magnitude of the velocity is represented by the intensity of red and blue colors.
  - Barriers are visualized as white regions.

## Customization

- **Grid Size**: You can change the grid size by modifying the `grid_l` and `grid_h` macros at the top of the code.
- **Barriers**: You can add or modify barriers by changing the initialization code in the `main` function.
- **Fluid Velocity**: Adjust the `fluidvel` variable to change the initial fluid velocity.

