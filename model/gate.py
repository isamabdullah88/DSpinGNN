import torch
from e3nn.nn import Gate
import torch.nn as nn
import e3nn.o3 as o3


class NonLinearGate(nn.Module):
    def __init__(self, l0dim, l1dim, l2dim):
        super(NonLinearGate, self).__init__()
        
        input_irreps = o3.Irreps(f"{l0dim}x0e + {l1dim}x1o + {l2dim}x2o")
        outlinear_irreps = o3.Irreps(f"{l0dim+l1dim+l2dim}x0e + {l1dim}x1o + {l2dim}x2o")

        self.linear = o3.Linear(input_irreps, outlinear_irreps)

        self.gate = Gate(
            f"{l0dim}x0e", [nn.SiLU()],
            f"{l1dim+l2dim}x0e", [nn.Sigmoid()],
            f"{l1dim}x1o + {l2dim}x2e"
        )

    def forward(self, node: torch.Tensor) -> torch.Tensor:
        node = self.linear(node)
        gated = self.gate(node)
        return gated
        