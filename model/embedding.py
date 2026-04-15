import torch
import torch.nn as nn
from e3nn import o3
from .convolution import ploynomial_cutoff
import e3nn.math as emath

class AtomEmbedding(nn.Module):
    def __init__(self, embeddim: int, l1dim: int, l2dim: int, numembeds: int = 118):
        super(AtomEmbedding, self).__init__()
        
        self.embedding = nn.Embedding(numembeds, embeddim)

        # self.input_irreps = o3.Irreps(f"{embeddim}x0e")
        # self.out_irreps = o3.Irreps(f"{embeddim}x0e + {l1dim}x1o + {l2dim}x2o")

        # self.linear = o3.Linear(self.input_irreps, self.out_irreps)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(z)

        # linear = self.linear(embeds)
        
        # return linear
        return embeds
    

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
    


class Radial(nn.Module):
    # Added min_dist to the initialization!
    def __init__(self, indim, outdim, rcut=4.1, min_dist=3.5):
        super(Radial, self).__init__()
        self.rcut = rcut
        self.min_dist = min_dist
        self.numbasis = indim

        self.model = nn.Sequential(
            nn.Linear(indim, 64),
            nn.SiLU(),
            nn.Linear(64, outdim)
        )

    def forward(self, dist):
        # 1. THE FIX: Pack all Bessel functions strictly between min_dist and rcut!
        # This guarantees ultra-high resolution for the 1NN phase transition.
        bessel = emath.soft_one_hot_linspace(
            dist, 
            start=self.min_dist,  # Changed from 0.0
            end=self.rcut, 
            number=self.numbasis, 
            basis='bessel', 
            cutoff=True
        )

        distf = self.model(bessel)

        # 2. Smooth envelope to ensure it hits 0.0 at rcut
        # Assuming you have your polynomial_cutoff imported
        cutoff = ploynomial_cutoff(dist, self.rcut).unsqueeze(-1)
        
        # 3. Masking out any impossible bonds that compress below min_dist
        # (Just in case you run extreme -15% strain simulations later)
        lower_mask = (dist >= self.min_dist).unsqueeze(-1).float()

        distf = distf * cutoff * lower_mask

        return distf.view(-1, self.numbasis)
    


class ConstrainedRadialEmbedding(nn.Module):
    def __init__(self, min_dist=3.5, max_dist=4.1, num_basis=128):
        super().__init__()
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_basis = num_basis
        
        # 1. Pack all 50 Gaussians tightly into this narrow window!
        # This gives you extreme ultra-high resolution for 1NN bonds.
        offset = torch.linspace(min_dist, max_dist, num_basis)
        self.register_buffer('offset', offset)
        
        # 2. Calculate the width so they overlap perfectly
        width = (offset[1] - offset[0]).item()
        self.coeff = -0.5 / (width ** 2)

    def forward(self, dist):
        # Gaussian Smearing
        diff = dist.view(-1, 1) - self.offset.view(1, -1)
        rbf = torch.exp(self.coeff * torch.pow(diff, 2))
        
        # 3. CRITICAL FIX: The Shifted Envelope
        # This maps min_dist -> 1.0 and max_dist -> 0.0 smoothly
        envelope_arg = torch.pi * (dist - self.min_dist) / (self.max_dist - self.min_dist)
        envelope = 0.5 * (torch.cos(envelope_arg) + 1.0)
        
        # 4. Strict Window Masking
        # Force exact zeros if a bond stretches beyond max_dist 
        # or compresses below min_dist
        mask = (dist >= self.min_dist) & (dist <= self.max_dist)
        envelope = envelope * mask.float()
        
        return rbf * envelope.view(-1, 1)