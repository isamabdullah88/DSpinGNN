import torch
import torch.nn as nn
from e3nn import o3

from .embedding import RadialEmbedding
    
class ExchangeBlock(nn.Module):
    def __init__(self, l0dim, l1dim, l2dim, numscalars=256, numbasis=128):
        super(ExchangeBlock, self).__init__()

        irrepsin = o3.Irreps(f"{l0dim}x0e + {l1dim}x1o + {l2dim}x2e")
        irrepsout = o3.Irreps(f"{numscalars}x0e")

        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in1=irrepsin, 
            irreps_in2=irrepsin, 
            irreps_out=irrepsout
        )

        self.normlayer = nn.LayerNorm(numscalars)

        self.rembedding = RadialEmbedding(cutoff=7.0, num_basis=numbasis)

        self.distfilter = nn.Sequential(
            nn.Linear(numbasis, 512),
            nn.SiLU(),
            nn.Linear(512, numscalars)
        )

        # UPGRADE: MLP now takes numscalars + 1 (for the explicit exponential distance feature)
        self.mlp_in = nn.Sequential(
            nn.Linear(numscalars + 1, 1024),
            nn.SiLU()
        )
        
        # UPGRADE: Residual block
        self.mlp_res = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.SiLU()
        )
        
        self.mlp_out = nn.Linear(1024, 1)

    def forward(self, nodes, batch):
        src, dst = batch.cr_edge_index
        graphidxs = batch.batch[src]
        
        cell = batch.cell.view(-1, 3, 3)
        bcell = cell[graphidxs]
        
        edge_shift = batch.cr_edge_shift.to(bcell.dtype)
        tvec = torch.einsum('ei, eij -> ej', edge_shift, bcell)
        
        radvec = batch.pos[dst] - batch.pos[src] + tvec
        dist = radvec.norm(dim=1, keepdim=False).view(-1, 1)

        # ==========================================
        # FIX 1: Explicit Permutation Symmetry
        # ==========================================
        mixed_forward = self.tp(nodes[src], nodes[dst])
        mixed_backward = self.tp(nodes[dst], nodes[src])
        mixed = 0.5 * (mixed_forward + mixed_backward)

        mixednorm = self.normlayer(mixed)

        distembedding = self.rembedding(dist)
        distfilter = self.distfilter(distembedding)

        regulated_feat = mixednorm * distfilter

        # ==========================================
        # FIX 2: Explicit Exponential Decay Feature
        # ==========================================
        # This explicitly tells the network when a bond is short-range (< 4.0 A)
        exp_dist = torch.exp(-dist) 
        
        # Concatenate the physics feature to the network features
        mlp_input = torch.cat([regulated_feat, exp_dist], dim=-1)

        # ==========================================
        # FIX 3: Residual MLP
        # ==========================================
        h1 = self.mlp_in(mlp_input)
        h2 = self.mlp_res(h1)
        h_out = h1 + h2  # Residual connection
        
        outx = self.mlp_out(h_out)

        return outx