# **Distributed Data Parallel (DDP) Training**

This guide explains how to train a CNN on the MNIST dataset using PyTorch's Distributed Data Parallel (DDP) framework. DDP is highly efficient for multi-GPU training as it reduces communication overhead compared to Data Parallel (DP).

---

## **Prerequisites**

1. Follow the steps in the [Single-GPU Training Guide](../02_singlegpu_training/) to set up your environment and repository.
2. Ensure the `workshop` Conda environment is activated.

```bash
$ module load miniconda
$ conda activate workshop
```
Follow download steps of DATA from single GPU given previously
---

## **Step 1: Adjustments for DDP**

### Key Changes from DP to DDP

#### Process Group Initialization
DDP requires initializing a process group:

```python
def setup():
    """Initialize the process group for DDP."""
    dist.init_process_group("nccl", init_method="env://")

def cleanup():
    """Destroy the process group after training."""
    dist.destroy_process_group()
```

#### Distributed Data Sampler
Use `DistributedSampler` to evenly distribute data across GPUs:

```python
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=dist.get_world_size(), rank=rank
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
```

#### DDP Model Wrapping
Wrap the model with `DistributedDataParallel`:

```python
model = Net().to(rank)
ddp_model = DDP(model, device_ids=[rank])
```

---

## **Step 2: DDP Training Script**

The training script (`mnist_ddpmodel.py`) includes all necessary DDP configurations. Key sections include:

### Model Definition
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
```

### Training Loop
```python
def train(args, model, train_loader, optimizer, epoch, rank):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and rank == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                  f"Loss: {loss.item():.6f}")
```

---

## **Step 3: DDP Slurm Script**

Create a Slurm submission script (`slurm_submit.sh`) to configure and execute the DDP script:

```bash
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

```

Submit the job:

```bash
(workshop) $ sbatch slurm_submit.sh
```
```
#!/bin/bash
#SBATCH --job-name=mnist_multi     # Job name
#SBATCH --nodes=2                  # Number of nodes
#SBATCH --ntasks-per-node=2        # Number of tasks (one per GPU per node)
#SBATCH --gres=gpu:2               # Number of GPUs on each node
#SBATCH --cpus-per-task=1          # Number of CPU cores per task
#SBATCH --partition=gpu            # GPU partition
#SBATCH --output=logs_%j.out       # Output log file
#SBATCH --error=logs_%j.err        # Error log file
#SBATCH --time=00:20:00            # Time limit

# Define variables for distributed setup
nodes_array=($(scontrol show hostnames $SLURM_JOB_NODELIST))
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head node IP: $head_node_ip"
# Set environment variables for PyTorch distributed training
export MASTER_ADDR=$head_node_ip   # Set the master node IP address
export MASTER_PORT=12355           # Any available port
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_GPUS_ON_NODE))
export RANK=$SLURM_PROCID          # Rank of the current process
export LOGLEVEL=INFO               # Log level for debugging
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Log environment variables for debugging
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"

# Load required modules and activate Conda environment
module purge
module load miniconda
conda activate workshop

# Run the PyTorch script with torchrun
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    mnist_ddpmodel.py --epochs=5 --batch-size=128

```

```bash
(workshop) $ sbatch multinode.sh
```

---

## **Step 4: Monitor and Analyze**

### Analyze Profiling Results

Add profiling output to a log file to analyze execution time:

```bash
tail -f logs_<job_id>.out
```


---
1. Adjust `--cpus-per-task` to values like 2 and 4 analyze the time and resource utilization (`slurm_submit.sh`).
2. Adjust `--ntasks` to values based on number of gpus 1 or 2, depending on nodes running on and analyse time and resource utilization (`slurm_submit.sh`) 
---

### Why Move to Hybrid Models like Fully Sharded Data Parallel (FSDP)?

1. **Memory Efficiency:** FSDP shards both model parameters and optimizer states across GPUs, reducing memory consumption.
2. **Scalability:** FSDP scales better for extremely large models, making it suitable for modern deep learning workflows.
3. **Gradient Sharding:** Unlike DDP, FSDP shards gradients, further optimizing memory usage during backpropagation.

These benefits make FSDP a preferred choice over DDP for training large models on multi-node setups.

[Learn more about DDP on the official PyTorch website](https://pytorch.org/docs/stable/notes/ddp.html)

[Explore Fully Sharded Data Parallel (FSDP) Training](../05_multigpu_fsdp_training/)
