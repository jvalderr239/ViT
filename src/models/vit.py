from torch import nn

from src.models.attention import TransformerEncoder
from src.models.classification import ClassificationHead
from src.models.patch_embeddings import PatchEmbedding


class ViT(nn.Sequential):
    """
    Visual Transformer 

    source: https://jalammar.github.io/illustrated-transformer/
    """
    def __init__(
        self,
        channels: int = 3,
        patch_size: int = 16,
        emb_size: int = 768,
        img_size: int = 224,
        depth: int = 12,
        n_classes: int = 1000,
        **kwargs,
    ):
        super().__init__(
            PatchEmbedding(channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes),
        )
