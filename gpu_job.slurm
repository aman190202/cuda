#!/bin/bash
#SBATCH --job-name=gpu_job
#SBATCH --output=gpu_job_%j.out
#SBATCH --error=gpu_job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Load necessary modules
module load cuda/11.8
module load gcc/9.3.0

# Print GPU information
nvidia-smi

# Compile the CUDA program
nvcc add.cu -o add

# Run the program
./add 