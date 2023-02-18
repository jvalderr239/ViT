from torch import Tensor, mean, nn


class LambdaMeanLayer(nn.Module):
    def __init__(self, method="linear", kernel: int = 16):
        """
        Custom layer to generate image patches

        Keyword Arguments:
            kernel -- image patch size (default: {16})
        """
        super(LambdaMeanLayer, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return self.torch.mean(x, 1, True)


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            LambdaMeanLayer(), nn.LayerNorm(emb_size), nn.Linear(emb_size, n_classes)
        )
