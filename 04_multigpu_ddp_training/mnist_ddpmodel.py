import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Define the model
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

def setup():
    """Initialize the process group for DDP."""
    dist.init_process_group("nccl", init_method="env://")

def cleanup():
    """Destroy the process group after training."""
    dist.destroy_process_group()

def train(args, model, train_loader, optimizer, epoch, rank):
    """Training loop for one epoch."""
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

def main():
    parser = argparse.ArgumentParser(description="PyTorch DDP MNIST Example")
    parser.add_argument("--batch-size", type=int, default=64, help="Global batch size for training")
    parser.add_argument("--test-batch-size", type=int, default=1000, help="Input batch size for testing")
    parser.add_argument("--epochs", type=int, default=14, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.7, help="Learning rate step gamma")
    parser.add_argument("--log-interval", type=int, default=10, help="Batches to log training status")
    parser.add_argument("--save-model", action="store_true", help="Save the trained model")
    args = parser.parse_args()

    rank = int(os.environ["LOCAL_RANK"])  # Rank of the process
    setup()  # Initialize DDP
    torch.cuda.set_device(rank)

    # Calculate per-GPU batch size
    world_size = dist.get_world_size()
    per_gpu_batch_size = args.batch_size // world_size

    print(f"[INFO] Global batch size: {args.batch_size}, Per-GPU batch size: {per_gpu_batch_size}")

    # Dataset and DataLoader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=per_gpu_batch_size, sampler=train_sampler)

    # Model, optimizer, and learning rate scheduler
    model = Net().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = optim.Adadelta(ddp_model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(args, ddp_model, train_loader, optimizer, epoch, rank)
        scheduler.step()

    # Save the model (only from rank 0)
    if rank == 0 and args.save_model:
        torch.save(ddp_model.state_dict(), "mnist_ddp_model.pth")

    cleanup()  # Cleanup DDP

if __name__ == "__main__":
    main()
