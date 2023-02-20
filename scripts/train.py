import time

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from tqdm import tqdm

from src.utils.transforms import ImageTransform

from . import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_dataloaders(data: Dataset, train_val_test_split: float = 0.8, batch_size: int = 64)-> DataLoader:
  """
  Generate train, val, test dataloaders

  Arguments:
      data -- dataset from which to generate train, val, test dataloaders

  Returns:
      Dictionary with each dataloader
  """
  # Parameters
  kwargs = {'num_workers': 4, 'pin_memory': True} if DEVICE=='cuda' else {}
  kwargs = {**kwargs, **{'batch_size': batch_size, 'shuffle': True,}}
  # Fetch dataset and generate DataLoaders
  dataset_size = len(data)
  train_size =  int(train_val_test_split * dataset_size)
  val_size = test_size = int(1. - train_val_test_split) * (dataset_size - train_size) // 2
  train_set, val_set, test_set = torch.utils.data.random_split(data, [train_size, val_size, test_size])

  dataloaders = {
    dtype: DataLoader(
      data, 
      transform=ImageTransform(dtype), **kwargs
      ) for dtype, data in (("train", train_set), ("val", val_set), ("test", test_set))
  }
  return dataloaders


dataloaders = generate_dataloaders(data=config.dataset)

print("[INFO] training the network...")
startTime = time.time()

#for e in tqdm(range(config.NUM_EPOCHS)):
