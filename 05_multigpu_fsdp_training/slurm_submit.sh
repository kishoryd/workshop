#!/bin/bash -l
#SBATCH --job-name=finetune_job       # Job name
#SBATCH --output=%x-%j.out            # File to write stdout
#SBATCH --error=%x-%j.err	      # file to error
#SBATCH --nodes=1                     # Single node
#SBATCH --ntasks=2                    # One task
#SBATCH --partition=gpu		      # partition GPU
##SBATCH --reservation=hpcai      # Reservation incase of urgent requirement
##SBATCH --nodelist=rpgpu*        # Specify reservation GPU node name provided
#SBATCH --cpus-per-task=10            # Number of CPU cores per task
#SBATCH --gres=gpu:2                  # Number of GPUs
#SBATCH --time=02:00:00               # Run time limit (HH:MM:SS)
#SBATCH --partition=gpu               # Partition for GPUs

# Load the required modules
module purge
module load miniconda

# Activate the Conda environment
conda activate workshop

# Total batch size and per-device batch size
total_batch_size=${TOTAL_BATCH_SIZE:-8}  # Adjust as per your training needs
batch_size_per_device=${BATCH_SIZE_PER_DEVICE:-4}  # Adjust for GPU memory constraints

# Number of GPUs available
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')

# Compute gradient accumulation steps
gradient_accumulation_steps=$(($total_batch_size / $batch_size_per_device / $num_gpus))

# Log configuration for debugging
echo "Using $num_gpus GPUs"
echo "Total batch size: $total_batch_size"
echo "Batch size per device: $batch_size_per_device"
echo "Gradient accumulation steps: $gradient_accumulation_steps"

# Run the distributed training script
torchrun \
    --nnodes=1 \
    --nproc-per-node=$num_gpus \
    chess.py \
        --batch_size_per_device $batch_size_per_device \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        "$@"

