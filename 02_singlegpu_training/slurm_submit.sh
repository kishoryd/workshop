#!/bin/bash
#SBATCH --job-name=mnist_1GPU    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --partition=gpu          # Specify GPU partition for GPU Nodes
##SBATCH --reservation=hpcai      # Reservation incase of urgent requirement
##SBATCH --nodelist=rpgpu*        # Specify reservation GPU node name provided
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --output=logs_%j.out     # output logfile name
#SBATCH --error=logs_%j.err      # error logfile name
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)

# which gpu node was used
echo "Running on host of RUDRA" $(hostname)

module purge #remove unneccesary loaded modules

#load the module
module load miniconda

#activate the environment
conda activate gujcost_workshop

#Try both the option, See the performance

kernprof -o ${SLURM_JOBID}_${SLURM_CPUS_PER_TASK}.lprof -l mnist_model.py --epochs=5 --batch-size 512

#kernprof -o ${SLURM_JOBID}_${SLURM_CPUS_PER_TASK}.lprof -l mnist_model.py --epochs=5 --no-cuda
