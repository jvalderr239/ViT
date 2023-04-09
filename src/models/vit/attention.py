from typing import Optional

from torch import Tensor, einsum, finfo, float32, nn


class MultiHeadAttention(nn.Module):
    """
    ViT model archictecture from source: https://jalammar.github.io/illustrated-transformer/
    """

    def __init__(
        self, embed_size: int = 768, num_heads: int = 8, dropout: float = 0.0
    ) -> None:
        """_summary_

        Keyword Arguments:
            embed_size -- number of nodes to represent token (default: {768})
            num_heads -- number of transformer blocks (default: {8})
            dropout -- dropout percentage (default: {0.0})
        """
        super().__init__()
        self.emb_size = embed_size
        self.num_heads = num_heads
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(embed_size, embed_size)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # split keys, queries and values in num_heads
        b, n, hd = x.shape
        queries = self.queries(x).reshape(b, self.num_heads, n, hd // self.num_heads)
        keys = self.keys(x).reshape(b, self.num_heads, n, hd // self.num_heads)
        values = self.values(x).reshape(b, self.num_heads, n, hd // self.num_heads)

        # find score
        energy = einsum(
            "bhqd, bhkd -> bhqk", queries, keys
        )  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = finfo(float32).min
            energy.masked_fill(~mask, fill_value)

        ## sqrt(d_k)
        scaling = self.emb_size ** (1 / 2)

        ## attention equation
        att = nn.functional.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = einsum("bhal, bhlv -> bhav ", att, values)
        b, h, n, d = out.shape
        out = out.reshape(b, n, h * d)
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fully_connected_layer):
        """
        Residual block

        Arguments:
            fully_connected_layer -- layer to wrap residual compute around
        """
        super().__init__()
        self.fn = fully_connected_layer

    def forward(self, x, **kwargs):
        """
        Residual block allows you to add layer output with layer input
        """
        temp = x
        return self.fn(x, **kwargs) + temp


class FeedForwardBlock(nn.Sequential):
    def __init__(self, embed_size: int, expansion: int = 4, drop_p: float = 0.0):
        """
        MLP block for ViT

        Arguments:
            emb_size -- embedding size

        Keyword Arguments:
            expansion -- upsample factor (default: {4})
            drop_p -- dropout rate (default: {0.})
        """
        super().__init__(
            nn.Linear(embed_size, expansion * embed_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * embed_size, embed_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size: int = 768,
        drop_p: float = 0.0,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.0,
        **kwargs,
    ):
        """
        Transformer Encoder Block

        Keyword Arguments:
            emb_size -- Embedding size (default: {768})
            drop_p -- Dropout Rate (default: {0.0})
            forward_expansion -- Dilation factor for ForwardFeedBlock (default: {4})
            forward_drop_p -- Dropout Rate for ForwardFeedBlock (default: {0.0})
        """
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, **kwargs),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size, expansion=forward_expansion, drop_p=forward_drop_p
                    ),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
