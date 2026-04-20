import torch
import torch.nn as nn
from .embedding import CosineSmearing, ChebyshevAngleSmearing


class ExchangeMLP(nn.Module):
    def __init__(self, numbasis=32):
        super(ExchangeMLP, self).__init__()

        embed_dim = 6 * numbasis

        # 1. Embeddings & LayerNorm
        self.dist_smearing = CosineSmearing(start=3.5, stop=4.5, num_basis=numbasis)
        self.leg_smearing = CosineSmearing(start=2.5, stop=3.1, num_basis=numbasis)
        self.angle_smearing = ChebyshevAngleSmearing(max_degree=numbasis)

        self.normlayer = nn.LayerNorm(embed_dim)

        self.deltaexchange = DeltaExchangeModel(embed_dim, hiddendim=16)


    def forward(self, batch):
        src, dst = batch.cr_edge_index
        graphidxs = batch.batch[src]
        cell = batch.cell.view(-1, 3, 3)
        bcell = cell[graphidxs]
        edge_shift = batch.cr_edge_shift.to(bcell.dtype)
        tvec = torch.einsum('ei, eij -> ej', edge_shift, bcell)
        radvec = batch.pos[dst] - batch.pos[src] + tvec
        dist = radvec.norm(dim=1, keepdim=False).view(-1, 1)
        angles = batch.cr_cr_angles.view(-1, 1)
        cr_i_legs = batch.cr_i_bonds

        # Embeddings
        distembedding = self.dist_smearing(dist)
        angle_embedding = self.angle_smearing(angles)
        cri_bondembedding1 = self.leg_smearing(cr_i_legs[:, 0].view(-1, 1))
        cri_bondembedding2 = self.leg_smearing(cr_i_legs[:, 1].view(-1, 1))
        cri_bondembedding3 = self.leg_smearing(cr_i_legs[:, 2].view(-1, 1))
        cri_bondembedding4 = self.leg_smearing(cr_i_legs[:, 3].view(-1, 1))

        embeddings = torch.cat([
            distembedding, cri_bondembedding1, cri_bondembedding2, 
            cri_bondembedding3, cri_bondembedding4, angle_embedding
        ], dim=-1)
        normedembeddings = self.normlayer(embeddings)
        
        outx = self.deltaexchange(angles, cr_i_legs, normedembeddings)

        return outx.view(-1)



class AnalyticalExchange(nn.Module):
    def __init__(self):
        super(AnalyticalExchange, self).__init__()
        # Learnable physics parameters initialized with reasonable physical guesses
        self.A = nn.Parameter(torch.tensor([-5.0]))
        self.B = nn.Parameter(torch.tensor([2.0]))
        self.C = nn.Parameter(torch.tensor([1.0]))
        self.alpha = nn.Parameter(torch.tensor([2.0]))
        
        # Reference equilibrium Cr-I bond length in Angstroms
        self.register_buffer('l_ref', torch.tensor([2.80]))

    def forward(self, cos_theta, legs):
        # cos_theta: [N, 1] (Already the cosine of the angle)
        # legs: [N, 4] (The four Cr-I bond distances)
        
        # 1. The Angular Term (Goodenough-Kanamori rules)
        angular_term = self.A * (cos_theta**2) + self.B * cos_theta + self.C
        
        # 2. The Orbital Overlap Decay Term
        # Average the 4 leg lengths
        mean_legs = legs.mean(dim=1, keepdim=True)
        overlap_term = torch.exp(-self.alpha * (mean_legs - self.l_ref))
        
        # 3. Final Analytical Prediction
        return angular_term * overlap_term


class DeltaExchangeModel(nn.Module):
    def __init__(self, numembeds, hiddendim=64):
        super(DeltaExchangeModel, self).__init__()
        
        # 1. The Physics Baseline
        self.physics_base = AnalyticalExchange()
        
        # 2. The ML Residual Block (The tiny network to fix quantum deviations)
        self.ml_residual = nn.Sequential(
            nn.Linear(numembeds, hiddendim),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(hiddendim, 1)
        )
        nn.init.zeros_(self.ml_residual[3].weight)
        nn.init.zeros_(self.ml_residual[3].bias)

    def forward(self, cos_theta, legs, embeddings):
        # Get the rough prediction from pure physics
        j_base = self.physics_base(cos_theta, legs)
        
        # Get the tiny quantum correction from the neural network
        j_correction = self.ml_residual(embeddings)
        
        # The final prediction is the physics + the ML correction
        return (j_base + j_correction).view(-1)