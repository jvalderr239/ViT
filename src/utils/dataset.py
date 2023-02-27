
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset, Subset


def generate_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 64,
    device="cpu",
    **kwargs
) -> Dict[str, DataLoader]:
    """_summary_

    Arguments:
        train_dataset -- train dataset
        val_dataset -- val dataset
        test_dataset -- test dataset

    Keyword Arguments:
        batch_size -- dataset batch size (default: {64})
        device -- machine for training (default: {"cpu"})

    Returns:
        _description_
    """
    # Parameters
    dataset_kwargs = {"batch_size": batch_size, "shuffle": False if kwargs.get("sampler") else True}
    cuda_kwargs = {"num_workers": 4, "pin_memory": True} if device == "cuda" else {}
    dataset_kwargs.update(cuda_kwargs)
    # Fetch dataset and generate DataLoaders

    dataloaders = {
        dtype: DataLoader(data, **dataset_kwargs, **kwargs)  # type: ignore
        for dtype, data in (
            ("train", train_dataset),
            ("test", test_dataset),
            ("val", val_dataset),
        )
    }
    return dataloaders

class VITSubset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices, labels):
        self.dataset = Subset(dataset, indices)
        self.targets = labels
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)