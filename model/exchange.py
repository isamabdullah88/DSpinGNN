import torch
import torch.nn as nn
from .embedding import GaussianSmearing

# ==========================================
# NEW: The Residual Block Superhighway
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.5):
        super(ResidualBlock, self).__init__()
        # A hidden layer that preserves the exact same dimension
        self.layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # THE SKIP CONNECTION: Add the original input 'x' back to the processed output
        return x + self.layer(x)


class ExchangeBlock(nn.Module):
    def __init__(self, l0dim, l1dim, l2dim, numscalars=64, numbasis=10):
        super(ExchangeBlock, self).__init__()

        self.numbasis = numbasis
        self.cosinebasis = numbasis
        embed_dim = numbasis * 6 

        # 1. Embeddings & LayerNorm
        self.rembedding = GaussianSmearing(start=-3.2, stop=4.5, num_gaussians=self.numbasis)
        self.bondembedding = GaussianSmearing(start=2.6, stop=3.2, num_gaussians=self.numbasis)
        self.cosembedding = GaussianSmearing(start=-0.20, stop=0.10, num_gaussians=self.cosinebasis)
        self.normlayer = nn.LayerNorm(embed_dim)

        self.deltaexchange = DeltaExchangeModel()

        # 2. Z-Score Buffers
        # self.register_buffer('cr_cr_mean', torch.tensor([4.00]))
        # self.register_buffer('cr_cr_std', torch.tensor([0.16]))
        # self.register_buffer('cr_i_mean', torch.tensor([2.80]))
        # self.register_buffer('cr_i_std', torch.tensor([0.035]))
        # self.register_buffer('angle_mean', torch.tensor([-0.025]))
        # self.register_buffer('angle_std', torch.tensor([0.06]))

        # ==========================================
        # 3. MIXTURE OF EXPERTS (MoE) ARCHITECTURE
        # ==========================================

        # self.exparg = nn.Sequential(
        #     nn.Linear(1, 4),
        #     nn.SiLU(),
        #     nn.Linear(4, 1)
        # )

        # self.angleres = nn.Sequential(
        #     nn.Linear(1, 4),
        #     nn.SiLU(),
        #     nn.Linear(4, 1)
        # )
        
        # Expert 1: Handles "Normal" Bulk Physics
        # self.expert_normal = nn.Sequential(
        #     nn.Linear(embed_dim, 2),
        #     nn.SiLU(),
        #     # ResidualBlock(2, dropout=0.5), # <-- Added Residual Block
        #     nn.Linear(2, 1)
        # )
        
        # Expert 2: Handles "Extreme" Rattling Physics
        # self.expert_extreme = nn.Sequential(
        #     nn.Linear(embed_dim, 4),
        #     nn.SiLU(),
        #     # ResidualBlock(4, dropout=0.5), # <-- Added Residual Block
        #     nn.Linear(4, 1)
        # )

        # The Router (Traffic Cop)
        # self.router = nn.Sequential(
        #     nn.Linear(embed_dim, 2),
        #     nn.SiLU(),
        #     nn.Dropout(0.2), # <-- MOVED DROPOUT HERE (See explanation below)
        #     nn.Linear(2, 2),
        #     nn.Softmax(dim=-1) 
        # )

    def forward(self, nodes, batch):
        # ... [Your exact same geometric and Z-score calculations go here] ...
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

        # dist_z = (dist - self.cr_cr_mean) / self.cr_cr_std
        # expf = -torch.exp(-dist)
        # angles_z = (angles - self.angle_mean) / self.angle_std
        # leg1_z = (cr_i_legs[:, 0].view(-1, 1) - self.cr_i_mean) / self.cr_i_std
        # leg2_z = (cr_i_legs[:, 1].view(-1, 1) - self.cr_i_mean) / self.cr_i_std
        # leg3_z = (cr_i_legs[:, 2].view(-1, 1) - self.cr_i_mean) / self.cr_i_std
        # leg4_z = (cr_i_legs[:, 3].view(-1, 1) - self.cr_i_mean) / self.cr_i_std

        # Embeddings
        distembedding = self.rembedding(dist) 
        cosembedding = self.cosembedding(angles)
        cri_bondembedding1 = self.bondembedding(cr_i_legs[:, 0].view(-1, 1))
        cri_bondembedding2 = self.bondembedding(cr_i_legs[:, 1].view(-1, 1))
        cri_bondembedding3 = self.bondembedding(cr_i_legs[:, 2].view(-1, 1))
        cri_bondembedding4 = self.bondembedding(cr_i_legs[:, 3].view(-1, 1))

        # Final concatenated feature vector
        normedembeddings = self.normlayer(torch.cat([
            distembedding, cosembedding,
            cri_bondembedding1, cri_bondembedding2, 
            cri_bondembedding3, cri_bondembedding4
        ], dim=-1))

        # 1. Ask the Router
        # routing_weights = self.router(normedembeddings)
        
        # 2. Get predictions
        # pred_normal = self.expert_normal(normedembeddings)
        # pred_extreme = self.expert_extreme(normedembeddings)
        
        # 3. Combine
        # final_prediction = (routing_weights[:, 0:1] * pred_normal) + (routing_weights[:, 1:2] * pred_extreme)
        # exp = -torch.exp(self.exparg(-dist)) + self.angleres(angles)
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
    def __init__(self, mlp_hidden_dim=8):
        super(DeltaExchangeModel, self).__init__()
        
        # 1. The Physics Baseline
        self.physics_base = AnalyticalExchange()
        
        # 2. The ML Residual Block (The tiny network to fix quantum deviations)
        self.ml_residual = nn.Sequential(
            nn.Linear(60, mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden_dim, 1)
        )

    def forward(self, cos_theta, legs, embeddings):
        # Get the rough prediction from pure physics
        j_base = self.physics_base(cos_theta, legs)
        
        # Get the tiny quantum correction from the neural network
        j_correction = self.ml_residual(embeddings)
        
        # The final prediction is the physics + the ML correction
        return (j_base + j_correction).view(-1)