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
    n_classes,
    loss_fn,
    ) -> float:
    """
    Function to train model for one epoch by enumerating batches

    Arguments:
        epoch_index -- current epoch
        training_loader -- dataset to traverse once
        tb_writer -- writer to store iteration info
        optimizer -- training optimizer
        model -- model to train
        n_classes -- number of class labels
        loss_fn -- loss function

    Returns:
        Average loss
    """
    running_loss = 0.0
    last_loss = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        # Could also include class weights for class balancing
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
        outputs = outputs.squeeze(1)
        # Compute the loss and its gradients
        # Be sure to pass floatTensor targets for class probabilities
        # Otherwise, pass class indices as LongTensor
        one_hot_labels = F.one_hot(labels, num_classes = n_classes)
        loss = loss_fn(outputs, one_hot_labels.float())
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            log.info(f"batch {i+1} loss: {last_loss}")
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss
