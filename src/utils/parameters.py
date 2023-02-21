from functools import partial
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset


def generate_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 64,
    device="cpu",
) -> Dict[str, DataLoader]:
    """
    Generate train, val, test dataloaders

    Arguments:
        data -- dataset from which to generate train, val, test dataloaders

    Returns:
        Dictionary with each dataloader
    """
    # Parameters
    kwargs = {"batch_size": batch_size, "shuffle": True}
    cuda_kwargs = {"num_workers": 4, "pin_memory": True} if device == "cuda" else {}
    kwargs.update(cuda_kwargs)
    # Fetch dataset and generate DataLoaders

    dataloaders = {
        dtype: DataLoader(data, **kwargs)  # type: ignore
        for dtype, data in (
            ("train", train_dataset),
            ("test", test_dataset),
        )
    }
    return dataloaders


def warmup(
    optimizer,
    training_steps: int,
    warmup_steps: int,
):
    def warmup_wrapper(
        current_step: int,
        training_steps: int,
        warmup_steps: int,
    ):
        if current_step < warmup_steps:  # current_step / warmup_steps * base_lr
            return float(current_step / warmup_steps)
        # (num_training_steps - current_step) / (num_training_steps - warmup_steps) * base_lr
        return max(
            0.0,
            float(training_steps - current_step)
            / float(max(1, training_steps - warmup_steps)),
        )

    lambda_warmup = partial(
        warmup_wrapper, training_steps=training_steps, warmup_steps=warmup_steps
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_warmup)
