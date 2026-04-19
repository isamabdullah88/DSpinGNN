import torch
import torch.nn as nn
from .embedding import GaussianSmearing


class ExchangeBlock(nn.Module):
    def __init__(self, l0dim, l1dim, l2dim, numscalars=64, numbasis=10):
        super(ExchangeBlock, self).__init__()

        self.numbasis = numbasis
        self.cosinebasis = numbasis
        embed_dim = 5 * self.numbasis + self.cosinebasis

        # 1. Embeddings & LayerNorm
        self.rembedding = GaussianSmearing(start=-3.0, stop=3.0, num_gaussians=self.numbasis)
        self.bondembedding = GaussianSmearing(start=-3.0, stop=3.0, num_gaussians=self.numbasis)
        self.cosembedding = GaussianSmearing(start=-3.0, stop=3.0, num_gaussians=self.cosinebasis)
        self.normlayer = nn.LayerNorm(embed_dim)

        # 2. Z-Score Buffers
        self.register_buffer('cr_cr_mean', torch.tensor([4.00]))
        self.register_buffer('cr_cr_std', torch.tensor([0.16]))
        self.register_buffer('cr_i_mean', torch.tensor([2.80]))
        self.register_buffer('cr_i_std', torch.tensor([0.035]))
        self.register_buffer('angle_mean', torch.tensor([-0.025]))
        self.register_buffer('angle_std', torch.tensor([0.06]))

        # ==========================================
        # 3. MIXTURE OF EXPERTS (MoE) ARCHITECTURE
        # ==========================================
        
        # Expert 1: Handles "Normal" Bulk Physics
        # (Slightly larger capacity for the vast majority of data)
        self.expert_normal = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )
        
        # Expert 2: Handles "Extreme" Rattling Physics
        # (Smaller capacity to prevent memorization of noise)
        self.expert_extreme = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

        # The Router (Traffic Cop)
        # Outputs 2 probabilities: [Probability of Normal, Probability of Extreme]
        self.router = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 2),
            nn.Dropout(0.5),
            nn.Softmax(dim=-1) # Ensures the two probabilities always sum to 1.0
        )

    def forward(self, nodes, batch):
        # ... (Keep your exact same geometric calculations and Z-score code here) ...
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

        dist_z = (dist - self.cr_cr_mean) / self.cr_cr_std
        angles_z = (angles - self.angle_mean) / self.angle_std
        leg1_z = (cr_i_legs[:, 0].view(-1, 1) - self.cr_i_mean) / self.cr_i_std
        leg2_z = (cr_i_legs[:, 1].view(-1, 1) - self.cr_i_mean) / self.cr_i_std
        leg3_z = (cr_i_legs[:, 2].view(-1, 1) - self.cr_i_mean) / self.cr_i_std
        leg4_z = (cr_i_legs[:, 3].view(-1, 1) - self.cr_i_mean) / self.cr_i_std

        # Embeddings
        distembedding = self.rembedding(dist_z) 
        cosembedding = self.cosembedding(angles_z)
        cri_bondembedding1 = self.bondembedding(leg1_z)
        cri_bondembedding2 = self.bondembedding(leg2_z)
        cri_bondembedding3 = self.bondembedding(leg3_z)
        cri_bondembedding4 = self.bondembedding(leg4_z)

        # Final concatenated feature vector
        normedembeddings = self.normlayer(torch.cat([
            distembedding, cosembedding, 
            cri_bondembedding1, cri_bondembedding2, 
            cri_bondembedding3, cri_bondembedding4
        ], dim=-1))

        # ==========================================
        # STEP C: Route and Predict
        # ==========================================
        
        # 1. Ask the Router who should handle each bond (Shape: [Total_edges, 2])
        routing_weights = self.router(normedembeddings)
        
        # 2. Get the prediction from BOTH experts (Shape: [Total_edges, 1])
        pred_normal = self.expert_normal(normedembeddings)
        pred_extreme = self.expert_extreme(normedembeddings)
        
        # 3. Combine them using the Router's probabilities
        # If Router says [0.99, 0.01], it relies 99% on the normal expert.
        final_prediction = (routing_weights[:, 0:1] * pred_normal) + (routing_weights[:, 1:2] * pred_extreme)

        return final_prediction.view(-1)