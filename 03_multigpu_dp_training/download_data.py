import torchvision
import warnings

warnings.simplefilter("ignore")

# GPU Nodes doesn't have internet, download the dataset in advance

# Download and store the MNIST dataset
_ = torchvision.datasets.MNIST(root='data',
                               train=True,
                               transform=None,
                               target_transform=None,
                               download=True)

