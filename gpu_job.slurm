#!/bin/bash
#SBATCH --job-name=plumes
#SBATCH --output=compile/nvdb3.out
#SBATCH --error=compile/nvdb3.err
#SBATCH --time=12:00:00               
#SBATCH --partition=gpu               
#SBATCH --gres=gpu:1                  
#SBATCH --mem=24G                     
#SBATCH --cpus-per-task=4


# Load necessary modules

module load cuda/11.8
module load miniconda3/23.11.0s


source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

# Activate conda environment and install necessary compilers
conda activate graphics
module load  gcc/10.1.0-mojgbnp 
# conda install conda-forge::boost -y
# conda install conda-forge::tbb -y
# conda install anaconda::blosc -y
# module load gcc
# cd openvdb/nanovdb/nanovdb
# mkdir -p build
# cd build
# # In your SLURM script, replace the cmake line with:
# cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
#          -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
#          -DOpenVDB_DIR=$CONDA_PREFIX/lib/cmake/OpenVDB \
#          -DCMAKE_MODULE_PATH=$CONDA_PREFIX/lib/cmake/OpenVDB
# make -j $(nproc) && make install
# # rm -rf build
# mkdir build
# cd build

# # Use conda's CMake and GCC
# # cmake .. -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
# make install DESTDIR=$CONDA_PREFIX

# Compile the renderer


# latest compilation code
nvcc -std=c++17 -g -G main.cu src/vdb_reader.cu -o render \
    -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib \
    -Xlinker -rpath -Xlinker $CONDA_PREFIX/lib \
    -lopenvdb -ltbb -lblosc -lboost_system -lboost_iostreams -lboost_filesystem -w

echo "Compilation complete!"

# Now compile CUDA code
# nvcc main.cu -o render
# nvprof ./render lights/untitled.filecache1_v1.0026_lights.bin

# for file in lights/*.bin; do
#     if [ -f "$file" ]; then
#         echo "Processing $file..."
#         ./render "$file"
#     fi
# done




export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
# # for file in /users/aagar133/scratch/cuda/v1/*.vdb; do
# #     if [ -f "$file" ]; then
# #         echo "Processing $file..."
# #         ./render "$file"
# #     fi
# # done
./render v1/untitled.filecache1_v1.0067.vdb

# echo "All files processed!" 

# Run with CUDA-GDB
