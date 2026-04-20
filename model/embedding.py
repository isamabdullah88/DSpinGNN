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
    

class ChebyshevAngleSmearing(nn.Module):
    def __init__(self, max_degree=5):
        super(ChebyshevAngleSmearing, self).__init__()
        self.max_degree = max_degree

    def forward(self, x):
        # x is already cos(theta), shape [N, 1].
        
        # CRITICAL SAFETY CLAMP: 
        # Prevents floating point errors from pushing x beyond [-1, 1]
        x = torch.clamp(x, min=-0.99999, max=0.99999)
        
        # T_1(x) = x
        t_basis = [x] 
        
        if self.max_degree >= 2:
            # T_2(x) = 2x^2 - 1
            t_basis.append(2 * x**2 - 1)
            
        # T_n(x) = 2x * T_{n-1}(x) - T_{n-2}(x)
        for i in range(2, self.max_degree):
            t_next = 2 * x * t_basis[-1] - t_basis[-2]
            t_basis.append(t_next)
            
        # Concatenates into shape [N, max_degree]
        return torch.cat(t_basis, dim=-1)


class CosineSmearing(nn.Module):
    def __init__(self, start, stop, num_basis=5):
        super(CosineSmearing, self).__init__()
        self.start = start
        self.stop = stop
        
        # Frequencies: [1, 2, 3, ..., num_basis]
        self.register_buffer('freqs', torch.arange(1, num_basis + 1).float())

    def forward(self, x):
        # 1. Normalize x to exactly [0.0, 1.0] based on the specific physical bounds
        x_norm = (x - self.start) / (self.stop - self.start)
        
        # 2. Clamp it to prevent aliasing (wrapping around) from extreme outliers
        x_norm = torch.clamp(x_norm, 0.0, 1.0)
        
        # 3. Scale to [0, pi] and compute the cosine features
        return torch.cos(x_norm * torch.pi * self.freqs) 


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