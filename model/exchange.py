import torch
import torch.nn as nn
from e3nn import o3

from .embedding import Radial, CosineAngleEmbedding
    
class ExchangeBlock(nn.Module):
    def __init__(self, l0dim, l1dim, l2dim, numscalars=64, numbasis=256):
        super(ExchangeBlock, self).__init__()

        irrepsin = o3.Irreps(f"{l0dim}x0e + {l1dim}x1o + {l2dim}x2e")
        irrepsout = o3.Irreps(f"{numscalars}x0e")

        # self.tp = o3.FullyConnectedTensorProduct(
        #     irreps_in1=irrepsin, 
        #     irreps_in2=irrepsin, 
        #     irreps_out=irrepsout
        # )

        # self.normlayer = nn.LayerNorm(numscalars)

        self.hidden_dim = 512
        self.numbasis = numbasis
        self.cosinebasis = 256
        # self.rembedding = ConstrainedRadialEmbedding(min_dist=3.5, max_dist=4.1, num_basis=numbasis)
        self.rembedding = Radial(self.numbasis, min_dist=3.65, rcut=4.35)
        self.bondembedding = Radial(indim=numbasis, min_dist=2.55, rcut=3.00)
        self.cosembedding = CosineAngleEmbedding(self.cosinebasis, min_cos=-0.14, max_cos=0.04, gamma=10.0)

        # self.distfilter = nn.Sequential(
        #     nn.Linear(numbasis, 512),
        #     nn.SiLU(),
        #     nn.Linear(512, numscalars)
        # )
        self.normlayer = nn.LayerNorm(5*self.numbasis + self.cosinebasis)  # Normalizing the combined feature vector

        # UPGRADE: MLP now takes numscalars + 1 (for the explicit exponential distance feature)
        self.mlp_in = nn.Sequential(
            nn.Linear(5*self.numbasis + self.cosinebasis, 1024),
            # nn.Dropout(0.2),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            # nn.Dropout(0.2),
            nn.SiLU()
        )
        
        # UPGRADE: Residual block
        self.mlp_res = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.SiLU()
        )
        # # FIX: Zero-initialize the final layer of the residual block
        # nn.init.zeros_(self.mlp_res[-1].weight)
        # nn.init.zeros_(self.mlp_res[-1].bias)
        
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
        # mixed = self.tp(nodes[src], nodes[dst])
        # mixed_backward = self.tp(nodes[dst], nodes[src])
        # mixed = 0.5 * (mixed_forward + mixed_backward)
        # print(f"Mixed features: {mixed.squeeze().cpu()}")

        # mixednorm = self.normlayer(mixed)
        # print(f"Normalized mixed features: {mixednorm.squeeze().cpu()}")

        # Explicit distance feature that decays with distance, providing a strong inductive bias for physical interactions
        # exp_dist = torch.exp(-dist)
        # dist = dist / 3.7

        distembedding = self.rembedding(dist)  # Inverse distance embedding
        # distfilter = self.distfilter(distembedding)
        cosembedding = self.cosembedding(batch.cr_cr_angles.view(-1, 1))
        # minbondembedding = self.bondembedding(batch.avg_cr_min.view(-1, 1))
        # maxbondembedding = self.bondembedding(batch.avg_cr_max.view(-1, 1))
        cri_bondembedding1 = self.bondembedding(batch.cr_i_bonds[:, 0].view(-1, 1))
        cri_bondembedding2 = self.bondembedding(batch.cr_i_bonds[:, 1].view(-1, 1))
        cri_bondembedding3 = self.bondembedding(batch.cr_i_bonds[:, 2].view(-1, 1))
        cri_bondembedding4 = self.bondembedding(batch.cr_i_bonds[:, 3].view(-1, 1))

        normedembeddings = self.normlayer(torch.cat([distembedding, cosembedding, cri_bondembedding1, cri_bondembedding2, cri_bondembedding3, cri_bondembedding4], dim=-1))

        cosanglesstr = "Cosine angles:\n"
        for c in batch.cr_cr_angles.view(-1).squeeze().cpu().numpy():
            cosanglesstr += f" {c:.6f}"

        # print(f"Cosine angle values: {cosanglesstr}")
        
        cosembeddingstr = "Cosine angle embedding values:\n"
        for ce in cosembedding.view(-1, self.cosinebasis).detach().cpu().numpy():
            cosembeddingstr += f" {torch.argmax(torch.tensor(ce)).item()}"  # Index of the most activated basis function

        # print(f"Cosine angle embedding values: {cosembeddingstr}")

        # angle_feat = batch.cr_cr_angles.view(-1, 1)
        # cri_min_feat = batch.avg_cr_min.view(-1, 1)
        # cri_max_feat = batch.avg_cr_max.view(-1, 1)

        # 3. The Ultimate Physics Vector
        # This 53-dimensional vector contains every piece of geometric 
        # information that dictates magnetic exchange.
        # edge_features = torch.cat([
        #     distembedding, 
        #     cosembedding, 
        #     minbondembedding,
        #     maxbondembedding
        # ], dim=-1)

        # regulated_feat = torch.cat([mixednorm, edge_features], dim=-1)
        
        h1 = self.mlp_in(normedembeddings)

        hout = h1 + self.mlp_res(h1)
        
        outx = self.mlp_out(hout)

        """
        diststr = "Distances:\n"
        for d in dist.detach().squeeze().cpu().numpy():
            diststr += f" {d:.2f}"
        # print(diststr)

        embeddingstr = "Distance embedding values:\n"
        for ds in distembedding.view(-1, self.numbasis).detach().cpu().numpy():
            embeddingstr += f" {torch.argmax(torch.tensor(ds)).item()}"  # Index of the most activated basis function
        # print(embeddingstr)

        exchangestr = "Ground-truth exchange values:\n"
        for j in batch.y_exchange.view(-1).detach().cpu().numpy():
            exchangestr += f" {j:.4f}"
        # print(exchangestr)

        predstr = "Predicted exchange values:\n"
        for j in outx.view(-1).detach().cpu().numpy():
            predstr += f" {j:.4f}"
        # print(predstr)
        """

        return outx.view(-1)