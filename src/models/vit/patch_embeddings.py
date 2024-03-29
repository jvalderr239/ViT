from torch import Tensor, cat, nn, randn


class LambdaPatchLayer(nn.Module):
    def __init__(self, patch_method="conv", kernel: int = 16):
        """
        Custom layer to generate image patches

        Keyword Arguments:
            kernel -- image patch size (default: {16})
        """
        super().__init__()
        self.kernel = kernel
        self.patch_method = patch_method

    def forward(self, x: Tensor) -> Tensor:
        return self.reshape(x) if self.patch_method == "conv" else self.patch(x)

    def patch(self, img: Tensor) -> Tensor:
        """
        Slices the images into square patches
        Arguments:
            img -- input image shape (b, c, h, w)
            kernel -- size of patches

        Returns:
            Tensor of flattened square image patches
        """
        assert len(img.shape) == 4, "Invalid input image batch size"
        b, c, *_ = img.shape
        patches = img.unfold(2, self.kernel, self.kernel).unfold(
            3, self.kernel, self.kernel
        )
        return patches.reshape(b, -1, c * self.kernel * self.kernel)

    def reshape(self, conv_output: Tensor) -> Tensor:
        """
        Method to flatten image patches formed from conv layer
        Arguments:
            conv_output -- Image patches

        Returns:
            Flattened image patches
        """
        b, c2, h2, w2 = conv_output.shape
        return conv_output.reshape(b, h2 * w2, c2)


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

    def __init__(
        self,
        channels: int = 3,
        patch_size: int = 16,
        emb_size: int = 768,
        img_size: int = 224,
        patch_method: str = "conv",
    ):
        """
        Generate linear projection layer to map image patches to token embedding layer

        Arguments:
            channels -- Number of channels for image
            patch_size -- Dimension of square image patch
            emb_size -- size of token embeddings
            img_size -- shape of square image
            patch_method -- method to use for image patching
        """
        super().__init__()

        if patch_method not in ("linear", "conv"):
            raise ValueError(f"Unrecognized patching method: {patch_method}")

        self.patch_method = patch_method

        if self.patch_method == "conv":
            self.projection = nn.Sequential(
                ## authors use conv layer for performance gain that uses kernel size of patch
                ## apply convolution to each patch and then flatten
                nn.Conv2d(
                    channels, emb_size, kernel_size=patch_size, stride=patch_size
                ),
                LambdaPatchLayer(patch_method=self.patch_method),
            )

        elif self.patch_method == "linear":
            self.projection = nn.Sequential(
                LambdaPatchLayer(patch_method=self.patch_method, kernel=patch_size),
                nn.Linear(patch_size * patch_size * channels, emb_size),
            )

        self.cls_token = nn.Parameter(randn(1, 1, emb_size))
        ## positional embedding size: N_PATCHES + 1 (token), EMBED_SIZE
        self.positions = nn.Parameter(
            randn((img_size // patch_size) ** 2 + 1, emb_size)
        )

    def forward(self, img: Tensor) -> Tensor:
        """
        Forward pass for input image through this model

        Arguments:
            img -- Input image for model

        Returns:
            1D linear projection of image patches along with positional and class embeddings
        """
        assert (
            len(img.shape) == 4
        ), f"Expected 4D (batched) image input but got {len(img.shape)}D"

        b, _, _, _ = img.shape
        x = self.projection(img)

        # prepend the cls token to the input and add position embedding
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = cat([cls_tokens, x], dim=1) + self.positions
        return x
