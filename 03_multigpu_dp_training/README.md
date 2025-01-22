# **Multi-GPU Training**

This guide extends the single-GPU training setup to a multi-GPU environment, leveraging PyTorch's `DataParallel` module for parallelism across multiple GPUs. It also demonstrates how to adjust your training script and Slurm configuration for multi-GPU jobs.

---

## **Step 1: Follow Single-GPU Setup**

Follow the steps outlined in the [Single-GPU Training Guide](../02_singlegpu_training/) to set up the environment, clone the repository, and download the dataset.

---

## **Step 2: Transition to Multi-GPU**

Navigate to the multi-GPU directory:

```bash
$ cd workshop/03_multigpu_dp_training
```

This directory contains the modified script and configurations for multi-GPU training.

---

## **Step 3: Changes from Single-GPU to Multi-GPU Training**

### Key Adjustments for Multi-GPU Support

#### Enable Multi-GPU with `DataParallel`
Ensure the model uses multiple GPUs with PyTorch’s `DataParallel`:

```python
# Initialize the CNN model
model = Net()

# Enable multi-GPU support using DataParallel
if use_cuda and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = nn.DataParallel(model)

model = model.to(device)  # Move model to the appropriate device
```

#### Adjust the DataLoader
Increase the `batch_size` to better utilize GPU memory:

```python
train_kwargs = {'batch_size': args.batch_size}
test_kwargs = {'batch_size': args.test_batch_size}
if use_cuda:
    cuda_kwargs = {'num_workers': num_workers, 'pin_memory': True, 'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
```

---

## **Step 4: Slurm Script for Multi-GPU Training**

Create a Slurm submission script (`slurm_submit.sh`) to allocate multiple GPUs:

```bash
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
kernprof -o ${SLURM_JOBID}_${SLURM_CPUS_PER_TASK}.lprof -l mnist_multigpu.py --epochs=5
```

Submit the job:

```bash
(workshop) $ sbatch slurm_submit.sh
```

### Monitor GPU and System Performance
Monitor GPU utilization and system threads:

```bash
$ ssh <gpu_node>
$ watch -n 0.1 nvidia-smi
$ top -u <hpcusername>
```

---

## **Step 5: Analyze Profiling Data**

After the job completes, analyze the profiling results:

```bash
(workshop) $ python -m line_profiler -rmt <job_id>_<slurmtask>.lprof
```

---

## **Assignments**

1. Adjust the `--gres=gpu` option in the Slurm script to use 1, 2 and analyze performance.
2. Adjust `--cpus-per-task` to values like 2 whether the time and resource utilization changes(`slurm_submit.sh`).
3. Modify the `--batch-size` to 128 and 256 to observe the impact on training speed and GPU utilization.
4. Experiment with different optimizers like `Adam` or `SGD` to compare convergence speeds.

---

## **Summary**

Multi-GPU training is a powerful approach to scale deep learning workloads. By enabling `DataParallel` and profiling the training process, you can achieve significant performance improvements. Efficient resource utilization ensures faster convergence and better scalability.

[Go back to Single-GPU Training](../02_singlegpu_training/)

### Why Data Parallel (DP) is Less Popular Compared to Distributed Data Parallel (DDP)?
Data Parallel (DP) is less popular due to its limitations in scalability and inefficiencies when handling large-scale distributed training. DP relies on replicating the model on each GPU, which can lead to significant communication overhead and slower training speeds. In contrast, Distributed Data Parallel (DDP) minimizes inter-GPU communication overhead and ensures better resource utilization by distributing data and gradients efficiently across multiple GPUs, making it the preferred approach for modern large-scale deep learning applications.Multi-GPU training is a powerful approach to scale deep learning workloads. By enabling `DataParallel` and profiling the training process, you can achieve significant performance improvements. Efficient resource utilization ensures faster convergence and better scalability.

[Proceed to Distributed Data Parallel Training](../04_multigpu_ddp_training/)
