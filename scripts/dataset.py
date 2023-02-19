import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from utils.transforms import ImageTransform

device = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = {'num_workers': 4, 'pin_memory': True} if device=='cuda' else {}
batch_size_train, batch_size_test = 64, 12

train_loader = DataLoader(
  datasets.MNIST('/files/', train=True, download=True),
  batch_size=batch_size_train, transform=ImageTransform("train"), **kwargs)

test_loader = DataLoader(
    datasets.MNIST('files/', train=False, download=True),
    batch_size=batch_size_test, transform=ImageTransform("test"), **kwargs)