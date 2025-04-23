#!/bin/bash
#SBATCH --job-name=installation
#SBATCH --output=output/warp_run.out
#SBATCH --error=output/warp_run.err
#SBATCH --time=00:30:00               
#SBATCH --partition=gpu               
#SBATCH --gres=gpu:1                  
#SBATCH --mem=16G                     
#SBATCH --cpus-per-task=4        

# Load necessary modules

module load cuda/11.8
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

# Activate conda environment and install necessary compilers
conda activate graphics



# Now compile CUDA code
# nvcc main.cu -o render
# nvprof ./render lights/untitled.filecache1_v1.0026_lights.bin

# for file in lights/*.bin; do
#     if [ -f "$file" ]; then
#         echo "Processing $file..."
#         nvprof ./render "$file"
#     fi
# done

nvprof ./render untitled.filecache1_v1.0086_lights.bin

echo "All files processed!" 