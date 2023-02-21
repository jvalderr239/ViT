from torch import cuda
from torchvision import datasets

from src.utils.transforms import ImageTransform

# define the base path to the input dataset and then use it to derive
# the path to the input images and annotation CSV files
BASE_PATH = "../train_files/"
DATASET_PATH = BASE_PATH + "dataset/"
MODEL_PATH = BASE_PATH + "model/"
TRAIN_PATH = BASE_PATH + "train/"
# determine the current device and based on that set the pin memory
# flag
DEVICE = "cuda" if cuda.is_available() else "cpu"
PIN_MEMORY = DEVICE == "cuda"
# specify ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-4
NUM_EPOCHS = 2
BATCH_SIZE = 32
TRAINING_STEPS = 20
WARMUP_STEPS = 5
# define dataset
TRAIN_DATASET = datasets.EuroSAT(
    BASE_PATH,
    download=True,
    train=True,
    transform=ImageTransform("train"),
    extensions=".jpg",
)
TEST_DATASET = datasets.EuroSAT(
    BASE_PATH,
    download=True,
    train=False,
    transform=ImageTransform("test"),
    extensions=".jpg",
)
