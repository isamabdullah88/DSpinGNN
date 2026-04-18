import torch
import torch.nn as nn
from e3nn import o3
from .convolution import ploynomial_cutoff
import e3nn.math as emath
from e3nn.math import soft_one_hot_linspace # assuming this is where emath comes from

class AtomEmbedding(nn.Module):
    def __init__(self, embeddim: int, l1dim: int, l2dim: int, numembeds: int = 118):
        super(AtomEmbedding, self).__init__()
        
        self.embedding = nn.Embedding(numembeds, embeddim)

        self.input_irreps = o3.Irreps(f"{embeddim}x0e")
        self.out_irreps = o3.Irreps(f"{embeddim}x0e + {l1dim}x1o + {l2dim}x2o")

        self.linear = o3.Linear(self.input_irreps, self.out_irreps)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(z)

        linear = self.linear(embeds)
        
        return linear
    


# class Radial(nn.Module):
#     # Added min_dist to the initialization!
#     def __init__(self, indim, outdim, rcut=4.1, min_dist=3.5, hidden_dim=64):
#         super(Radial, self).__init__()
#         self.rcut = rcut
#         self.min_dist = min_dist
#         self.numbasis = indim

#         self.model = nn.Sequential(
#             nn.Linear(indim, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, outdim)
#         )

#     def forward(self, dist):
#         # 1. THE FIX: Pack all Bessel functions strictly between min_dist and rcut!
#         # This guarantees ultra-high resolution for the 1NN phase transition.
#         bessel = emath.soft_one_hot_linspace(
#             dist, 
#             start=self.min_dist,  # Changed from 0.0
#             end=self.rcut, 
#             number=self.numbasis, 
#             basis='bessel', 
#             cutoff=True
#         )

#         distf = self.model(bessel)

#         # 2. Smooth envelope to ensure it hits 0.0 at rcut
#         # Assuming you have your polynomial_cutoff imported
#         cutoff = ploynomial_cutoff(dist, self.rcut).unsqueeze(-1)
        
#         # 3. Masking out any impossible bonds that compress below min_dist
#         # (Just in case you run extreme -15% strain simulations later)
#         lower_mask = (dist >= self.min_dist).unsqueeze(-1).float()

#         distf = distf * cutoff * lower_mask

#         return distf.view(-1, self.numbasis)

class Radial(nn.Module):
    def __init__(self, indim, rcut=4.5, min_dist=3.5):
        super(Radial, self).__init__()
        self.rcut = rcut
        self.min_dist = min_dist
        self.numbasis = indim

    def forward(self, dist):
        # 1. Pack all Bessel functions strictly between min_dist and rcut
        bessel = soft_one_hot_linspace(
            dist, 
            start=self.min_dist, 
            end=self.rcut, 
            number=self.numbasis, 
            basis='bessel', 
            cutoff=True
        )

        # 2. Smooth envelope to ensure it hits 0.0 at rcut
        # (Assuming your polynomial_cutoff function is imported/defined)
        cutoff = ploynomial_cutoff(dist, self.rcut).unsqueeze(-1)
        
        # 3. Masking out any impossible bonds that compress below min_dist
        lower_mask = (dist >= self.min_dist).unsqueeze(-1).float()

        # Multiply the raw bessel basis directly by the masks
        distf = bessel * cutoff * lower_mask

        return distf.view(-1, self.numbasis)
    


class CosineAngleEmbedding(nn.Module):
    def __init__(self, indim, min_cos=-0.20, max_cos=0.10, gamma=10.0):
        super().__init__()
        self.numbasis = indim
        
        # 1. The High-Resolution Gaussian Expansion
        offset = torch.linspace(min_cos, max_cos, indim)
        self.register_buffer('offset', offset)
        
        # Calculate the distance between centers
        width = (offset[1] - offset[0]).item()
        
        # 2. THE FIX: The 'gamma' factor makes the Gaussians "skinnier"
        # This prevents massive overlap and forces sharper bin activation!
        self.coeff = (-0.5 / (width ** 2)) * gamma

    def forward(self, cos_angle):
        diff = cos_angle.view(-1, 1) - self.offset.view(1, -1)
        gaussians = torch.exp(self.coeff * torch.pow(diff, 2))
        
        # Mask out extreme outliers so they don't light up the edge bins infinitely
        mask = (cos_angle >= self.offset[0]) & (cos_angle <= self.offset[-1])
        gaussians = gaussians * mask.float().view(-1, 1)
        
        # 3. THE FIX: Return the raw Gaussians directly!
        return gaussians.view(-1, self.numbasis)