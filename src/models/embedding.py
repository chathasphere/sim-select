import torch
import math
from torch.nn import Module, Linear, ReLU, TransformerEncoder, \
    TransformerEncoderLayer, LayerNorm
from typing import Callable, Sequence


class MLP(torch.nn.Sequential):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Sequence[int] = (64, 64),
        activation: Callable[[], Module] = None,
        normalize: bool = False,
    ):
        if activation is None:
            activation = ReLU

        normalization = LayerNorm if normalize else lambda: None

        layers = []

        for before, after in zip(
            (in_features, *hidden_features),
            (*hidden_features, out_features),
        ):
            layers.extend([
                Linear(before, after),
                activation(),
                normalization(),
            ])

        layers = layers[:-2]
        layers = filter(lambda layer: layer is not None, layers)

        super().__init__(*layers)

        self.in_features = in_features
        self.out_features = out_features
        

 
class TransformerEmbedding(Module):
    def __init__(self, d_x: int, d_emb: int, nhead: int = 8, 
                 n_blocks: int = 1, dropout: float = 0):
        super().__init__()
        self.embed = Linear(d_x, d_emb)
        self.pos_encode = PositionalEncoding(d_emb)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_emb, nhead=nhead, dropout=dropout,
            batch_first=True, dim_feedforward=4*d_emb)
        norm = LayerNorm(d_emb)
        self.transformer = TransformerEncoder(encoder_layer, n_blocks, norm)
    
    
    def forward(self, x):
        # x should have shape (batch, seq, feature)
        # may need to transpose
        x = self.embed(x)
        x = self.pos_encode(x)
        x = self.transformer(x)
        # output should have shape (batch, )
        return x.mean(dim=1)



class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=1000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)