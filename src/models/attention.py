from torch import nn


class MultiHeadAttention(nn.Module):
  def __init__(self, embed_size: int = 512, num_heads: int = 8, dropout: float = 0.) -> None:
    super().__init__()