# Volumetric Path-tracer illuminated by million lights in CUDA


## Demo Videos

1. **100,000 Lights:**  
   Rendering volumetric explosion illuminated by 100,000 point lights using GPU-parallel sampling.  
   ![100k Lights](https://github.com/user-attachments/assets/e42190d2-ab9a-48d2-b1dd-a94354d8e0bb)

2. **500 Lights (Baseline):**  
   Same scene with only 500 lights to demonstrate the visual difference and performance baseline.  
   ![500 Lights](https://github.com/user-attachments/assets/16b13a97-edcf-4353-a0c7-894c338c2185)

3. **Light Point Visualization:**  
   Point lights shown as emissive spheres without volume scattering to visualize spatial distribution.  
   ![Light Points](https://github.com/user-attachments/assets/fbfe76f6-40db-47c5-85a6-75527c90293c)


A high-performance volumetric path-tracer implemented in CUDA C++ for rendering realistic explosions and volumetric effects. This renderer is capable of handling millions of light sources efficiently through GPU acceleration.


## Features
- Volumetric path tracing for realistic smoke and explosion rendering
- Support for OpenVDB data format
- Efficient handling of millions of light sources
- GPU-accelerated rendering using CUDA
- High-quality volumetric scattering and absorption

## Requirements
- CUDA 11.8
- GCC 10.1.0
- OpenVDB library
- Conda environment with the following packages:
  - boost
  - tbb
  - blosc

## Installation

1. Set up your conda environment:
```bash
conda create -n graphics python=3.8
conda activate graphics
conda install -c conda-forge boost tbb blosc
```

2. Install OpenVDB:
```bash
conda install -c conda-forge openvdb
```

## Building

Compile the renderer using the following command:
```bash
nvcc -std=c++17 main.cu src/vdb_reader.cu -o render \
     -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib \
     -Xlinker -rpath -Xlinker $CONDA_PREFIX/lib \
     -lopenvdb -ltbb -lblosc -lboost_system -lboost_iostreams -lboost_filesystem -w
```

## Running

1. Set up the required environment variables:
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
```

2. Run the renderer with a VDB file:
```bash
./render path/to/your/file.vdb
```

## Project Structure
- `main.cu`: Main CUDA implementation
- `src/vdb_reader.cu`: OpenVDB data loading and processing
- `v1/`: Directory containing VDB files for rendering
