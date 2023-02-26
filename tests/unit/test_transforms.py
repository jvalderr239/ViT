from unittest import TestCase
from src.utils.transforms import ImageTransform
import numpy as np
from torch import Tensor


np.random.seed(239)
SHAPE = (555, 777, 3)
class TryTesting(TestCase):
    def test_transform_shape(self):
        img = np.random.randint(low=0, high=255, size=SHAPE, dtype=np.uint8)
        for name in ("train", "val", "test"):
            transformed_img = ImageTransform(name)(img)
            assert isinstance(transformed_img, Tensor)
            transformed_img = transformed_img.unsqueeze(0).repeat(5, 1, 1, 1) # add batch dim
            assert transformed_img.shape == (5, 3, 224, 224)
