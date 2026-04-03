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
        # 1. Gaussian Smearing (The Embedding)
        # diff shape: [num_edges, num_basis]
        diff = dist.view(-1, 1) - self.offset.view(1, -1)
        rbf = torch.exp(self.coeff * torch.pow(diff, 2))
        
        # 2. Cosine Envelope
        # Starts at 1, smoothly goes to 0 at the cutoff
        envelope = 0.5 * (torch.cos(dist * torch.pi / self.cutoff) + 1.0)
        
        # Force exact zero beyond cutoff just in case
        envelope = envelope * (dist < self.cutoff).float()
        
        # 3. Multiply embedding by the envelope
        return rbf * envelope.view(-1, 1)
    

class ExchangeBlock(nn.Module):
    def __init__(self, l0dim, l1dim, l2dim):
        super(ExchangeBlock, self).__init__()

        irrepsin = o3.Irreps(f"{l0dim}x0e + {l1dim}x1o + {l2dim}x2e")
        # irrepsout = o3.Irreps(f"{l0dim}x0e")
        out_scalars = 64
        irrepsout = o3.Irreps(f"{out_scalars}x0e")

        # self.linear = o3.Linear(irrepsin, irrepsout)
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in1=irrepsin, 
            irreps_in2=irrepsin, 
            irreps_out=irrepsout
        )

        numbasis = 64
        self.rembedding = RadialEmbedding(cutoff=7.0, num_basis=numbasis)

        self.mlp = nn.Sequential(
            nn.Linear(out_scalars + numbasis, 64),
            nn.SiLU(),
            # nn.Linear(256, 128),
            # nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )


    def forward(self, nodes, batch):

        # print('nodes: ', nodes.shape)
        # print('cr_edge_index: ', batch.cr_edge_index.shape)
        # print('cr_edge_shift: ', batch.cr_edge_shift.shape)

        src, dst = batch.cr_edge_index

        # neighbors = nodes[src]

        graphidxs = batch.batch[src]
        # print('graphidxs: ', graphidxs.shape)
        # print('graphidxs: ', graphidxs)
        
        cell = batch.cell.view(-1, 3, 3)
        
        bcell = cell[graphidxs]
        # print('bcell: ', bcell.shape)
        
        edge_shift = batch.cr_edge_shift.to(bcell.dtype)

        tvec = torch.einsum('ei, eij -> ej', edge_shift, bcell)
        
        radvec = batch.pos[dst] - batch.pos[src] + tvec

        dist = radvec.norm(dim=1, keepdim=False)
        
        # l0feat_src = self.linear(nodes[src])
        # l0feat_dst = self.linear(nodes[dst])

        mixed = self.tp(nodes[src], nodes[dst])
        # print('l0feat_src: ', l0feat_src.shape)
        # print('l0feat_dst: ', l0feat_dst.shape)
        # print('dist: ', dist.shape)
        distembedding = self.rembedding(dist)

        inx = torch.cat([mixed, distembedding], dim=1)
        # print('inx: ', inx.shape)

        outx = self.mlp(inx)
        # print('outx: ', outx.shape)

        return outx