from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    """
    The standard Transformer receives as input a 1D sequence of token embeddings.
    To handle 2D images, reshape the input image of shape (H, W, C) into a
      sequence of flattened 2D patches of size N x (P^2 x C) where
      (H, W) is the shape of the original image,
      C is the number of channels,
      (P, P) is the shape of each image patch and
      N = HW/P^2 is the number of patches

    Source: https://arxiv.org/pdf/2010.11929.pdf
    """

    def __init__(self, channels: int = 3, patch_size: int = 16, emb_size: int = 768):
        """
        Generate linear projection layer to map image patches to token embedding layer

        Args:
            channels: Number of channels for image
            patch_size: Dimension of square image patch
            emb_size: size of token embeddings
        """
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            ## authors use conv layer for performance gain that uses kernel size of patch
            ## apply convolution to each patch and then flatten
            nn.Conv2d(channels, emb_size, kernel_size=patch_size, stride=patch_size),
        )

    def forward(self, img: Tensor) -> Tensor:
        """
        Forward pass
        """
        return self.projection(img)
