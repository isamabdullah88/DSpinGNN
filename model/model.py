import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool

from .embedding import AtomEmbedding
from .interaction import InteractionBlock   
from .outblock import OutputBlock


def force(energy, pos):
    ones = torch.ones_like(energy)

    grads = torch.autograd.grad(outputs=energy, inputs=pos, grad_outputs=ones, create_graph=True,
                                retain_graph=True, allow_unused=True)[0]

    forces = -grads
    return forces


class NequIP(nn.Module):
    def __init__(self, mps=False):
        super(NequIP, self).__init__()
        
        self.numembeds = 50
        self.l0dim = 32
        self.l1dim: int = 16
        self.l2dim: int = 8

        self.atomembeds = AtomEmbedding(self.l0dim, self.l1dim, self.l2dim, self.numembeds)

        self.interaction_block1 = InteractionBlock(self.l0dim, self.l1dim, self.l2dim, mps=mps)

        self.interaction_block2 = InteractionBlock(self.l0dim, self.l1dim, self.l2dim, mps=mps)
        
        self.interaction_block3 = InteractionBlock(self.l0dim, self.l1dim, self.l2dim, mps=mps)

        # self.interaction_block4 = InteractionBlock(self.l0dim, self.l1dim, self.l2dim, mps=mps)

        # self.interaction_block5 = InteractionBlock(self.l0dim, self.l1dim, self.l2dim, mps=mps)

        self.output_block = OutputBlock(self.l0dim, self.l1dim, self.l2dim)

    def forward(self, batch) -> torch.Tensor:

        nodes = self.atomembeds(batch.z)

        # print('z: ', z)
        interacted1 = self.interaction_block1(nodes, batch)

        interacted2 = self.interaction_block2(interacted1, batch)

        interacted3 = self.interaction_block3(interacted2, batch)

        # interacted4 = self.interaction_block4(interacted3, batch)

        # interacted5 = self.interaction_block5(interacted4, batch)

        output = self.output_block(interacted3, batch.z)

        energyt = global_add_pool(output, batch.batch)

        return energyt
    

