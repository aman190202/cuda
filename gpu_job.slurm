#!/bin/bash
#SBATCH --job-name=warp_run
#SBATCH --output=output/warp_run.out
#SBATCH --error=output/warp_run.err
#SBATCH --time=02:00:00               # Adjust as needed
#SBATCH --partition=gpu               # Or the appropriate GPU partition
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --mem=16G                     # Adjust memory as needed
#SBATCH --cpus-per-task=4            # Adjust CPU cores as needed

# Load necessary modules
module load cuda/11.8
module load gcc/10.1
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

# Print GPU information
# g++ add.cpp -o add

conda activate graphics
conda install conda-forge::boost -y
conda install conda-forge::tbb -y
conda install anaconda::blosc -y
cd openvdb
mkdir build
cd build
cmake ..
make -j4 && make install
nvcc main.cu -o render
nvprof ./render

# Your command to run the GPU application
# Example: python your_gpu_script.py 