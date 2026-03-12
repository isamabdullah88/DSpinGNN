
import torch.nn as nn
from e3nn import o3
from torch_geometric.nn import radius_graph

class Exchange(nn.Module):
    def __init__(self, l0dim, l1dim, l2dim):
        super(Exchange, self).__init__()

        # irrepsin = o3.Irreps(f"{l0dim}x0e + {l1dim}x1o + {l2dim}x2e")
        # irrepsout = o3.Irreps(f"{l0dim}x0e")

        # self.linear = o3.Linear(irrepsin, irrepsout)
        self.embedding = nn.Embedding(100, l0dim)

        self.mlp = nn.Sequential(
            nn.Linear(l0dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        

    def forward(self, z, batch, pos):
        nodes = self.embedding(z)
        
        radius_graph(pos, r=8.0, batch=batch)
        
        x = x.squeeze(-1)
        x = self.mlp(x)

        return x