import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool

from .embedding import AtomEmbedding
from .interaction import InteractionBlock   
from .outblock import OutputBlock
from .exchange import ExchangeBlock


def calcforce(energy, pos):
    ones = torch.ones_like(energy)

    grads = torch.autograd.grad(outputs=energy, inputs=pos, grad_outputs=ones, create_graph=True,
                                retain_graph=True, allow_unused=False)[0]

    forces = -grads
    return forces


class DSpinGNN(nn.Module):
    def __init__(self):
        super(DSpinGNN, self).__init__()
        
        self.numembeds = 118
        self.l0dim = 32
        self.l1dim: int = 16
        self.l2dim: int = 8
        self.rcut = 7.0

        self.atomembeds = AtomEmbedding(self.l0dim, self.l1dim, self.l2dim, self.numembeds)

        self.interaction_block1 = InteractionBlock(self.l0dim, self.l1dim, self.l2dim, self.rcut)

        self.interaction_block2 = InteractionBlock(self.l0dim, self.l1dim, self.l2dim, self.rcut)
        
        self.interaction_block3 = InteractionBlock(self.l0dim, self.l1dim, self.l2dim, self.rcut)

        # self.interaction_block4 = InteractionBlock(self.l0dim, self.l1dim, self.l2dim, self.rcut)

        # self.interaction_block5 = InteractionBlock(self.l0dim, self.l1dim, self.l2dim, self.rcut)

        self.exchange_block = ExchangeBlock(self.l0dim, self.l1dim, self.l2dim)

        self.output_block = OutputBlock(self.l0dim, self.l1dim, self.l2dim)

    def forward(self, batch) -> tuple[torch.Tensor, torch.Tensor]:

        nodes = self.atomembeds(batch.z)

        interacted1 = self.interaction_block1(nodes, batch)

        interacted2 = self.interaction_block2(interacted1, batch)

        interacted3 = self.interaction_block3(interacted2, batch)

        # interacted4 = self.interaction_block4(interacted3, batch)

        # interacted5 = self.interaction_block5(interacted4, batch)
        
        exchangej = self.exchange_block(interacted3, batch)

        output = self.output_block(interacted3, batch.z)

        energyt = global_add_pool(output, batch.batch)

        return energyt, exchangej
    

