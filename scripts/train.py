import logging
import logging.config
import time
from datetime import datetime
from pathlib import Path

import config
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.vit import ViT
from src.utils.parameters import generate_dataloaders, warmup
from src.utils.train_utils import train_one_epoch

# setup logger
logging.config.fileConfig('logging.conf')
log = logging.getLogger('train')

log.info(f"Generating {config.BASE_PATH}")
Path(config.BASE_PATH).mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataloaders = generate_dataloaders(
    train_dataset=config.TRAIN_DATASET,
    val_dataset=config.VAL_DATASET,
    test_dataset=config.TEST_DATASET,
    batch_size=config.BATCH_SIZE,
    device=DEVICE,
)

log.info("Retrieving train parameters...")
startTime = time.time()

# training parameters
model = ViT(n_classes=config.NUM_CLASSES)
# Optimizers specified in the torch.optim package
optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), weight_decay=0.1)
lr_scheduler = warmup(
    optimizer=optimizer,
    training_steps=config.TRAINING_STEPS,
    warmup_steps=config.WARMUP_STEPS,
)
loss_fn = torch.nn.CrossEntropyLoss()

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(
    f"{config.TRAIN_PATH}runs/EUROSAT_{timestamp}"
)
best_vloss = 1_000_000.0

log.info(f"Training the network on {config.__DATASET_SIZE__} images")
for epoch_number in tqdm(range(config.NUM_EPOCHS)):
    log.info(f"EPOCH {epoch_number + 1}:")

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(
        epoch_index=epoch_number,
        training_loader=dataloaders.get("train"),
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        tb_writer=writer,
        n_classes=config.NUM_CLASSES
    )

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(dataloaders["val"]):
        vinputs, vlabels = vdata
        voutputs = model(vinputs)

        predicted_class_idx = voutputs.argmax(0)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (len(dataloaders["test"]) + 1)
    writer.add_text(f"LOSS: train {avg_loss} valid {avg_vloss}")
    
    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars(
        "Training vs. Validation Loss",
        {"Training": avg_loss, "Validation": avg_vloss},
        epoch_number + 1,
    )
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f"{config.MODEL_PATH}model_{timestamp}_{epoch_number}"
        torch.save(model.state_dict(), model_path)

running_tloss = 0.0
predicted_classes = []
for i, tdata in enumerate(dataloaders["test"]):
    tinputs, tlabels = tdata
    toutputs = model(tinputs)

    predicted_class_idx = toutputs.argmax(0)
    tloss = loss_fn(toutputs, tlabels)
    running_tloss += tloss
    predicted_classes.append(predicted_class_idx)

avg_tloss = running_tloss / (len(dataloaders["test"]) + 1)
writer.add_text(f"LOSS: test {avg_tloss}")
writer.add_scalars(predicted_classes)
# Log the running loss averaged per batch
# for both training and validation
writer.add_scalars(
    "Test_Loss",
    {"Test": avg_tloss},
)
writer.flush()
