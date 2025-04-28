#!/bin/bash
#SBATCH --job-name=installation
#SBATCH --output=output/nvdb.out
#SBATCH --error=output/nvdb.err
#SBATCH --time=02:00:00               
#SBATCH --partition=gpu               
#SBATCH --gres=gpu:1                  
#SBATCH --mem=16G                     
#SBATCH --cpus-per-task=12  

# Load necessary modules

module load cuda/11.8
module load miniconda3/23.11.0s

source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

# Activate conda environment and install necessary compilers
conda activate graphics
# conda install conda-forge::boost -y
# conda install conda-forge::tbb -y
# conda install anaconda::blosc -y
module load gcc
cd openvdb/nanovdb/nanovdb
mkdir -p build
cd build
# In your SLURM script, replace the cmake line with:
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
         -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
         -DOpenVDB_DIR=$CONDA_PREFIX/lib/cmake/OpenVDB \
         -DCMAKE_MODULE_PATH=$CONDA_PREFIX/lib/cmake/OpenVDB
make -j $(nproc) && make install
# rm -rf build
# mkdir build
# cd build

# # Use conda's CMake and GCC
# # cmake .. -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
# make install DESTDIR=$CONDA_PREFIX

# Compile the renderer
cd ../../../../
nvcc main.cu -o render \
    -I$CONDA_PREFIX/include \
    -I$CONDA_PREFIX/usr/local/include \
    -L$CONDA_PREFIX/lib \
    -L$CONDA_PREFIX/usr/local/lib64 \
    -lopenvdb -ltbb -lz -lblosc -lboost_iostreams -lboost_system -lboost_filesystem

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


# nvcc main.cu -o render
# nvprof ./render untitled.filecache1_v1.0086_lights.bin




# cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
#          -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
#          -DOpenVDB_DIR=$CONDA_PREFIX/usr/local/lib64/cmake/OpenVDB \
#          -DOpenVDB_INCLUDE_DIR=$CONDA_PREFIX/usr/local/include \
#          -DOpenVDB_LIBRARY_DIR=$CONDA_PREFIX/usr/local/lib64 \
#          -DNANOVDB_BUILD_TOOLS=ON \
#          -DNANOVDB_BUILD_CONVERT=ON \
#          -DNANOVDB_USE_OPENVDB=ON

echo "All files processed!" 