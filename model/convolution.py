
import torch.nn as nn
import e3nn.math as emath
from e3nn import o3
from torch_geometric.nn import radius_graph
from torch_geometric.utils import scatter


def gen_instructions(irreps_in, irreps_edge, irreps_out):
    instructions = []

    # Iterate over all possible combinations
    for i, (mul_in, ir_in) in enumerate(irreps_in):
        for j, (mul_edge, ir_edge) in enumerate(irreps_edge):
            for k, (mul_out, ir_out) in enumerate(irreps_out):
                
                # 1. Check Physics (Angular Momentum + Parity)
                if ir_out in ir_in * ir_edge:
                    
                    # 2. Check if "Depthwise" is possible
                    # 'uvu' requires the input channel count to match the output
                    if mul_in == mul_out:
                        # Valid Depthwise connection!
                        instructions.append((i, j, k, 'uvu', True))

    # print(f"Generated {len(instructions)} valid instructions.")
    # print(instructions)
    return instructions

def ploynomial_cutoff(x, rcut):

    p = 6
    envelope = 1 - (p * (x / rcut)**5) + ((p - 1) * (x / rcut)**6)

    mask = (x < rcut).to(x.dtype)
    return envelope * mask


class Radial(nn.Module):
    def __init__(self, indim, outdim, rcut):
        super(Radial, self).__init__()
        self.rcut = rcut
        self.numbasis = indim

        self.model = nn.Sequential(
            nn.Linear(indim, 64),
            nn.SiLU(),
            nn.Linear(64, outdim)
        )

    def forward(self, dist):
        bessel = emath.soft_one_hot_linspace(dist, 0.0, 5.0, self.numbasis, basis='bessel', cutoff=True)

        distf = self.model(bessel)

        cutoff = ploynomial_cutoff(dist, self.rcut).unsqueeze(-1)

        distf = distf * cutoff

        return distf

        
class Convolution(nn.Module):
    def __init__(self, l0dim, l1dim, l2dim, numbasis=20, rcut=5.0, mps=True):
        super(Convolution, self).__init__()
        self.mps = mps

        self.avg_neighbors = 25.0
        
        self.numbasis = numbasis
        self.rcut = rcut

        irreps_in1 = o3.Irreps(f"{l0dim}x0e + {l1dim}x1o + {l2dim}x2e")
        irreps_in2 = o3.Irreps("1x0e + 1x1o + 1x2e")
        irreps_out = o3.Irreps(f"{l0dim}x0e + {l1dim}x1o + {l2dim}x2e")

        self.linear1 = o3.Linear(irreps_in1, irreps_in1)

        instructions = gen_instructions(irreps_in1, irreps_in2, irreps_out)

        self.tp = o3.TensorProduct(
            irreps_in1 = irreps_in1,  # input features
            irreps_in2 = irreps_in2,   # spherical harmonics l=1
            irreps_out = irreps_out,  # output features
            instructions = instructions,
            internal_weights = False,
            shared_weights = False
        )

        self.linear2 = o3.Linear(irreps_out, irreps_out)

        numweights = self.tp.weight_numel
        self.radialMLP = Radial(self.numbasis, numweights, self.rcut)


    def forward(self, nodes, pos, batch):

        if self.mps:
            edgeidxs = radius_graph(pos.cpu(), r=self.rcut, batch=batch.cpu(), max_num_neighbors=100).to("mps")
        else:
            edgeidxs = radius_graph(pos, r=self.rcut, batch=batch, max_num_neighbors=100)

        src, dst = edgeidxs

        neighbors = nodes[src]

        posvec = pos[src] - pos[dst]

        dist = posvec.norm(dim=1, keepdim=False)

        radial = self.radialMLP(dist)

        ylm = o3.spherical_harmonics(l=[0, 1, 2], x=posvec, normalize=True, normalization='component')

        mxdneighbors = self.linear1(neighbors)

        messages = self.tp(mxdneighbors, ylm, weight=radial)

        mxdmessages = self.linear2(messages)

        aggregated = scatter(mxdmessages, dst, dim=0, reduce='add')

        aggregated = aggregated / (self.avg_neighbors**0.5)

        return aggregated