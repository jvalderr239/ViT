import logging

import torch.nn.functional as F
from torch.utils.data import DataLoader

# setup logger
logging.config.fileConfig('logging.conf')
log = logging.getLogger('train')


def train_one_epoch(
    epoch_index: int, 
    training_loader: DataLoader, 
    tb_writer, 
    optimizer, 
    model, 
    loss_fn,
    n_classes
    ):
    running_loss = 0.0
    last_loss = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
        outputs = outputs.squeeze(1)
        log.info(f"Output shape: {outputs.shape}")
        # Compute the loss and its gradients
        #one_hot_labels = F.one_hot(labels, num_classes = n_classes).long()
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print(f"  batch {i+1} loss: {last_loss}")
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss
