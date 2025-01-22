# **Training with Model Parallelism via Fully Sharded Data Parallel (FSDP)**

This guide provides an overview and example for training using Fully Sharded Data Parallel (FSDP), which enables efficient memory utilization and scalability for large models.

---

## **Background**

FSDP shreds both model parameters and optimizer states across GPUs, reducing memory overhead and improving performance. For more background, refer to:

- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Everything about Distributed Training and Efficient Finetuning](https://sumanthrh.com/post/distributed-and-efficient-finetuning/)
- [Hugging Face FSDP Documentation](https://huggingface.co/docs/accelerate/usage_guides/fsdp)

---

## **FSDP Training Example**

In this example, we fine-tune a [CodeLlama-7b](https://huggingface.co/codellama/CodeLlama-7b-hf) model on a dataset of [chess moves](https://huggingface.co/datasets/laion/strategic_game_chess).

This demo is only meant to illustrate a simple and transparent training run with FSDP, and should not be used as a deep-learning training script. We intentially omit common features such as model checkpoints, evaluation, etc. Most pytorch training libraries support FSDP out-of-the-box.

### Steps:

1. **Environment Setup:**
   Install the required packages in a Conda environment:
   ```bash
   $ module load miniconda
   $ conda activate workshop

   ```

2. **Download Models:**
   Since compute nodes may not have internet access, pre-download the models:
   ```bash
   (workshop) $ python downloadmodel.py
   ```

3. **FSDP Wrapping:**
   See the script `chess.py` for implementation details on how to wrap model layers with FSDP.

4. **Run Training:**
   Submit the `slurm_submit.sh` Slurm script.

---

## **Comparison: DDP vs. FSDP**

| Feature                        | Distributed Data Parallel (DDP) | Fully Sharded Data Parallel (FSDP) |
|-------------------------------|----------------------------------|------------------------------------|
| **Memory Usage**               | Higher                          | Optimized with sharding           |
| **Model Scalability**          | Limited to GPU memory           | Suitable for large-scale models   |
| **Parameter Sharding**         | No                              | Yes                                |
| **Gradient Accumulation**      | Standard                        | Optimized                         |

FSDP improves upon DDP by sharding model parameters and gradients, enabling the training of much larger models.

---

## **FSDP Slurm Script**

Here is the Slurm script (`slurm_submit.sh`) for running FSDP training:

```bash
#!/bin/bash -l
#SBATCH --job-name=finetune_job       # Job name
#SBATCH --output=%x-%j.out            # File to write stdout
#SBATCH --error=%x-%j.err             # File to write stderr
#SBATCH --nodes=1                     # Single node
#SBATCH --ntasks=2                    # One task per GPU
#SBATCH --partition=gpu               # Partition for GPUs
#SBATCH --cpus-per-task=10            # Number of CPU cores per task
#SBATCH --gres=gpu:2                  # Number of GPUs
#SBATCH --time=02:00:00               # Run time limit (HH:MM:SS)

# Load required modules
module purge
module load miniconda

# Activate the Conda environment
conda activate workshop

# Batch size configuration
TOTAL_BATCH_SIZE=8
BATCH_SIZE_PER_DEVICE=4
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')
GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / BATCH_SIZE_PER_DEVICE / NUM_GPUS))

# Log configurations
echo "Using $NUM_GPUS GPUs"
echo "Total batch size: $TOTAL_BATCH_SIZE"
echo "Batch size per device: $BATCH_SIZE_PER_DEVICE"
echo "Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"

# Run the training script
torchrun \
    --nnodes=1 \
    --nproc-per-node=$NUM_GPUS \
    chess.py \
        --batch_size_per_device $BATCH_SIZE_PER_DEVICE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        "$@"
```

Submit the job:

```bash
$ sbatch slurm_submit.sh
```

---

## **FSDP Tuning Tips**

### Batch Sizes
- `batch_size_per_device`: Sequences processed in one pass per GPU.
- Total optimization batch size: `batch_size_per_device * num_gpus * gradient_accumulation_steps`.

### Gradient Checkpointing
- Enable with the `--gradient_checkpointing` flag to save memory.

### Performance Questions
1. Increase `batch_size_per_device` by factors of 2 and observe the effect on time per gradient step and After a certain threshold, the batch size approaches the memory limits of the GPU.
2. Add gradient checkpointing and measure training speed for various batch sizes.
   Gradient Checkpointing Tradeoff:
      Pros: Allows larger batch sizes and supports training larger models.
      Cons: Introduces a slight computational overhead, increasing training time per step.

---

## **Summary**

FSDP enables efficient training of large models by sharding parameters and gradients across GPUs. It outperforms DDP in scalability and memory optimization.

[Return to Single-GPU Training Guide](../02_singlegpu_training/)

[Go to Distributed Data Parallel (DDP) Training Guide](../04_multigpu_ddp_training/)
