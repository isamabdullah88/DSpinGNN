import torch
import torch.nn as nn
from e3nn import o3

class AtomEmbedding(nn.Module):
    def __init__(self, embeddim: int, l1dim: int, l2dim: int):
        super(AtomEmbedding, self).__init__()
        
        numembeds = 100
        self.embedding = nn.Embedding(numembeds, embeddim)

        self.input_irreps = o3.Irreps(f"{embeddim}x0e")
        self.out_irreps = o3.Irreps(f"{embeddim}x0e + {l1dim}x1o + {l2dim}x2o")

        self.linear = o3.Linear(self.input_irreps, self.out_irreps)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(z)

        linear = self.linear(embeds)
        
        return linear