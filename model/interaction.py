import torch
import torch.nn as nn
from e3nn import o3
from .convolution import Convolution
from .gate import NonLinearGate

class InteractionBlock(nn.Module):
    def __init__(self, l0dim, l1dim, l2dim, mps=False):
        super(InteractionBlock, self).__init__()

        # self-interaction
        in_irreps = o3.Irreps(f"{l0dim}x0e + {l1dim}x1o + {l2dim}x2o")
        self.owninteraction1 = o3.Linear(in_irreps, in_irreps)

        self.owninteraction2 = o3.Linear(in_irreps, in_irreps)

        self.convolution = Convolution(l0dim, l1dim, l2dim, mps=mps)

        self.gate = NonLinearGate(l0dim, l1dim, l2dim)

    def forward(self, nodes: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:

        self_interacted1 = self.owninteraction1(nodes)

        convolved = self.convolution(self_interacted1, pos, batch)

        self_interacted2 = self.owninteraction2(convolved)

        mixed = nodes + self_interacted2

        gated = self.gate(mixed)

        return gated