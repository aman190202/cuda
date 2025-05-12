# Volumetric Path-tracer illuminated by million lights in CUDA

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
