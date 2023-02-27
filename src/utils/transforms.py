import random
from typing import Union

import albumentations as A
from albumentations.pytorch import ToTensorV2
from numpy import array, generic, ndarray
from PIL.Image import Image
from torch import Tensor

random.seed(239)

train_transforms = A.Compose(
    [
        A.RandomRotate90(p=0.4),
        A.Flip(p=0.3),
        A.Transpose(p=0.15),
        A.GaussNoise(p=0.4),
        A.OneOf(
            [
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ],
            p=0.2,
        ),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf(
            [
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.PiecewiseAffine(p=0.3),
            ],
            p=0.2,
        ),
        A.OneOf(
            [
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ],
            p=0.3,
        ),
        A.HueSaturationValue(p=0.3),
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

val_transforms = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

class ImageTransform:
    DEFAULT_TRANSFORMS = {
        "train": train_transforms,
        "val": val_transforms,
        "test": val_transforms,
    }
    def __init__(self, datatype: str) -> None:
        self.datatype = datatype

    def __call__(self, img: Union[Tensor, Image, ndarray]):
        
        if isinstance(img, Tensor):
            img = img.numpy()
        elif isinstance(img, Image):
            img = array(img)
        elif not isinstance(img, (ndarray, generic)):
            raise ValueError(f"Expected input of type Tensor but received {type(img)}")

        return self.DEFAULT_TRANSFORMS[self.datatype](image=img)["image"]
