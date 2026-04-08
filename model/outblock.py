import torch
import torch.nn as nn
import e3nn.o3 as o3

class OutputBlock(nn.Module):
    def __init__(self, l0dim, l1dim, l2dim, num_species=100):
        super(OutputBlock, self).__init__()
        
        input_irreps = o3.Irreps(f"{l0dim}x0e + {l1dim}x1o + {l2dim}x2e")
        
        self.linear = o3.Linear(input_irreps, o3.Irreps("1x0e"))

        self.scale = nn.Parameter(torch.tensor(1.0))
        self.shift = nn.Parameter(torch.tensor(torch.zeros(num_species)))

    def forward(self, nodes: torch.Tensor, atom: torch.Tensor) -> torch.Tensor:
        rawe = self.linear(nodes)

        baseline = self.shift[atom].view(-1, 1)

        atome = (rawe * self.scale) + baseline

        return atome.squeeze()