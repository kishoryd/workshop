#!/bin/bash
#SBATCH --job-name=mnist_multiGPU  # Job name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=2                 # Number of tasks
#SBATCH --cpus-per-task=6          # Number of CPU cores per task
#SBATCH --partition=gpu            # GPU partition
##SBATCH --reservation=hpcai      # Reservation incase of urgent requirement
##SBATCH --nodelist=rpgpu*        # Specify reservation GPU node name provided
#SBATCH --gres=gpu:2               # Number of GPUs (adjust as needed)
#SBATCH --output=logs_%j.out       # Output log file
#SBATCH --error=logs_%j.err        # Error log file
#SBATCH --time=00:20:00            # Time limit

# Log the node and GPUs being used
echo "Running on host $(hostname)"

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Load required modules
module purge
module load miniconda

# Activate the Conda environment
conda activate workshop

# Run the script
kernprof -o ${SLURM_JOBID}_${SLURM_CPUS_PER_TASK}.lprof -l mnist_multigpu.py --epochs=5 --batch-size=128

