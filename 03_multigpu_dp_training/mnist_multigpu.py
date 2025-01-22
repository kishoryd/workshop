import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# Define the CNN model

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

# Training loop
@profile  # Decorator for profiling the training process
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()  # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):  # Iterate through the training batches
        data, target = data.to(device), target.to(device)  # Move data and targets to the device
        optimizer.zero_grad()  # Clear previous gradients
        output = model(data)  # Forward pass
        loss = F.nll_loss(output, target)  # Compute the negative log-likelihood loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters
        if batch_idx % args.log_interval == 0:  # Print progress at specified intervals
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Testing loop
def test(model, device, test_loader):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():  # Disable gradient calculation for testing
        for data, target in test_loader:  # Iterate through the test batches
            data, target = data.to(device), target.to(device)  # Move data and targets to the device
            output = model(data)  # Forward pass
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # Sum batch losses
            pred = output.argmax(dim=1, keepdim=True)  # Get predicted classes
            correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')

# Main function
def main():
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',  # Training batch size
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',  # Testing batch size
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',  
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',  
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, 
                        help='For Saving the current Model')
    args = parser.parse_args()

    # Check for GPU availability
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Configure data loader settings
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    num_workers = int(os.getenv('SLURM_CPUS_PER_TASK', 1))
    if use_cuda:
        cuda_kwargs = {'num_workers': num_workers, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Data preprocessing: normalization and tensor conversion
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)


    # Try using Adam optimizer for faster convergence:
    #optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Try using Exponential decay of the learning rate:
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # Print dataset sizes
    print(f"Train set size: {len(dataset1)}")
    print(f"Test set size: {len(dataset2)}")

    # Initialize the CNN model
    model = Net()

    # Enable multi-GPU support using DataParallel if multiple GPUs are available
    if use_cuda and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    model = model.to(device)  # Move the model to the appropriate device (CPU/GPU)

    # Define optimizer and learning rate scheduler
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Training and testing loop
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()  # Update learning rate as per scheduler

    # Save the trained model
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")  # Save model weights

if __name__ == '__main__':
    main()

