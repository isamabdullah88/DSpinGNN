import torch
import torch.nn as nn
from e3nn import o3
from torch_geometric.nn import radius_graph

class RadialEmbedding(nn.Module):
    def __init__(self, cutoff=7.0, num_basis=50):
        super().__init__()
        self.cutoff = cutoff
        self.num_basis = num_basis
        
        # Create 50 evenly spaced centers between 0 and the cutoff
        offset = torch.linspace(0.0, cutoff, num_basis)
        self.register_buffer('offset', offset)
        
        # Calculate the width of the Gaussians so they overlap smoothly
        width = (offset[1] - offset[0]).item()
        self.coeff = -0.5 / (width ** 2)

    def forward(self, dist):
        # Gaussian Smearing (The Embedding)
        # diff shape: [num_edges, num_basis]
        diff = dist.view(-1, 1) - self.offset.view(1, -1)
        rbf = torch.exp(self.coeff * torch.pow(diff, 2))
        
        # Cosine Envelope
        # Starts at 1, smoothly goes to 0 at the cutoff
        envelope = 0.5 * (torch.cos(dist * torch.pi / self.cutoff) + 1.0)
        
        # Force exact zero beyond cutoff just in case
        envelope = envelope * (dist < self.cutoff).float()
        
        # Multiply embedding by the envelope
        return rbf * envelope.view(-1, 1)
    

class ExchangeBlock(nn.Module):
    def __init__(self, l0dim, l1dim, l2dim, numscalars=128, numbasis=64):
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
            nn.Linear(numbasis, 128),
            nn.SiLU(),
            nn.Linear(128, numscalars)
        )

        self.mlp = nn.Sequential(
            nn.Linear(numscalars, 512),
            nn.SiLU(),
            nn.Linear(512, 1)
        )


    def forward(self, nodes, batch):
        # Distance computation using edge indices and periodic boundary conditions
        src, dst = batch.cr_edge_index

        graphidxs = batch.batch[src]
        
        cell = batch.cell.view(-1, 3, 3)
        bcell = cell[graphidxs]
        
        edge_shift = batch.cr_edge_shift.to(bcell.dtype)

        tvec = torch.einsum('ei, eij -> ej', edge_shift, bcell)
        
        radvec = batch.pos[dst] - batch.pos[src] + tvec

        dist = radvec.norm(dim=1, keepdim=False).view(-1, 1)

        # Fully connected tensor product to mix features from source and destination nodes
        mixed = self.tp(nodes[src], nodes[dst])

        # Normalize the mixed features. This is important to ensure that the scale of the features
        # is consistent before applying the distance-based filter.
        mixednorm = self.normlayer(mixed)

        distembedding = self.rembedding(dist)
        distfilter = self.distfilter(distembedding)

        regulated_feat = mixednorm * distfilter

        outx = self.mlp(regulated_feat)

        return outx