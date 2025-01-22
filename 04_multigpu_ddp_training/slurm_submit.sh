#!/bin/bash
#SBATCH --job-name=mnist_multi     # Job name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=2                 # Number of tasks (one per GPU)
#SBATCH --gres=gpu:2               # Number of GPUs on the node
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
##SBATCH --reservation=hpcai      # Reservation incase of urgent requirement
##SBATCH --nodelist=rpgpu*        # Specify reservation GPU node name provided
#SBATCH --partition=gpu            # GPU partition
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

# Set environment variables for DDP
export MASTER_ADDR=localhost       # Use localhost for single node
export MASTER_PORT=12355           # Any available port
export WORLD_SIZE=$SLURM_NTASKS    # Total number of processes (tasks)
export RANK=$SLURM_PROCID          # Rank of the current process

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}


# Log environment variables for debugging
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"

# Run the script with kernprof
torchrun --nproc_per_node=2 mnist_ddpmodel.py --epochs=5 --batch-size=128

