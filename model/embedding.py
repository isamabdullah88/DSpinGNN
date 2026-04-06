import torch
import torch.nn as nn
from e3nn import o3

class AtomEmbedding(nn.Module):
    def __init__(self, embeddim: int, l1dim: int, l2dim: int, numembeds: int = 100):
        super(AtomEmbedding, self).__init__()
        
        self.embedding = nn.Embedding(numembeds, embeddim)

        self.input_irreps = o3.Irreps(f"{embeddim}x0e")
        self.out_irreps = o3.Irreps(f"{embeddim}x0e + {l1dim}x1o + {l2dim}x2o")

        self.linear = o3.Linear(self.input_irreps, self.out_irreps)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(z)
        # print('embeds: ', embeds.shape)

        linear = self.linear(embeds)
        
        return linear
    

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