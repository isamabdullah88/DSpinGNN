import torch
import torch.nn as nn
from e3nn import o3
from torch_geometric.nn import radius_graph

class ExchangeBlock(nn.Module):
    def __init__(self, l0dim, l1dim, l2dim):
        super(ExchangeBlock, self).__init__()

        irrepsin = o3.Irreps(f"{l0dim}x0e + {l1dim}x1o + {l2dim}x2e")
        irrepsout = o3.Irreps(f"{l0dim}x0e")

        self.linear = o3.Linear(irrepsin, irrepsout)

        self.mlp = nn.Sequential(
            nn.Linear(2*l0dim+1, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
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
        
        l0feat_src = self.linear(nodes[src])
        l0feat_dst = self.linear(nodes[dst])
        # print('l0feat_src: ', l0feat_src.shape)
        # print('l0feat_dst: ', l0feat_dst.shape)
        # print('dist: ', dist.shape)

        inx = torch.cat([l0feat_src, l0feat_dst, dist.view(-1, 1)], dim=1)
        # print('inx: ', inx.shape)

        outx = self.mlp(inx)
        # print('outx: ', outx.shape)

        return outx