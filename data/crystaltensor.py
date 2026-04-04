import torch
import numpy as np

from .crI3 import CrI3

class CrystalGraphTensor:
    def __init__(self):
        pass
        # self.CrI3 = CrI3()

        # self.pos = torch.tensor(self.CrI3.batoms.get_positions(), dtype=torch.float32)
        # self.cell = torch.tensor(self.CrI3.cell, dtype=torch.float32)


    def tensorgraph(self, rcut, atoms):
        """
        Builds a graph with periodic boundary conditions from scratch including ghost atoms using PyTorch tensors
        """
        N = atoms.get_global_number_of_atoms()
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        cell = torch.tensor(atoms.get_cell(), dtype=torch.float32)
        
        mx, my = torch.meshgrid(torch.tensor([-1, 0, 1], dtype=torch.int64), torch.tensor([-1, 0, 1], dtype=torch.int64), indexing='ij')
        mz = torch.zeros_like(mx)
        # print('mx: ', mx.dtype, mx.shape)
        # print('my: ', my.dtype, my.shape)
        # print('mz: ', mz.dtype, mz.shape)

        stack = torch.stack((mx.flatten(), my.flatten(), mz.flatten()), axis=1)
        # print('stack: ', stack.dtype, stack.shape)
        # print('stack: ', [tuple(row.tolist()) for row in stack])
        
        translations = torch.mm(stack.to(torch.float32), cell)
        # print('\npositions: \n', [tuple(row.tolist()) for row in pos])
        # print('\ntranslations: \n', [tuple(row.tolist()) for row in translations])

        ghostpos = pos.unsqueeze(0) + translations.unsqueeze(1)
        ghostpos = ghostpos.view(-1, 3)
        # print('\nghostpos: \n', [tuple(row.tolist()) for row in ghostpos])

        dist = torch.cdist(ghostpos, pos)
        # print('\ndist: ', dist.dtype, dist.shape)

        mask = (dist > 1e-4) & (dist <= rcut)

        ghostidxs, atomidxs = torch.where(mask)
        # print('\nindex matrix: \n', [(i, j) for i, j in zip(ghostidxs.tolist(), atomidxs.tolist())])

        srcidxs = atomidxs
        dstidxs = ghostidxs % N
        cellidxs = ghostidxs // N
        # print('\nsrcidxs: ', srcidxs)
        # print('\ndstidxs: ', dstidxs)
        # print('\ncellidxs: ', cellidxs)
        # 4. EXPLICITLY DELETE SELF-LOOPS (Same atom AND Same Cell)
        # If src == dst AND it's in the home cell (cell shift is [0,0,0])
        is_home_cell = (stack[cellidxs] == 0).all(dim=1)
        is_self_loop = (srcidxs == dstidxs) & is_home_cell

        # Invert to keep only valid edges
        valid_edges = ~is_self_loop

        # Apply the filter
        srcidxs = srcidxs[valid_edges]
        dstidxs = dstidxs[valid_edges]
        cellidxs = cellidxs[valid_edges]

        # print('\nedges: ', edges.dtype, edges.shape)
        # print('\nedges: ', [tuple(row.tolist()) for row in edges])
        shifts = stack[cellidxs]
        # print('\nshifts: ', [tuple(row.tolist()) for row in shifts])
        edges = torch.stack((srcidxs, dstidxs), axis=1)

        return edges.t(), shifts
    


if __name__ == "__main__":
    from data.testedges import verify_against_ase
 
    z = torch.tensor([24, 24, 51, 51, 51, 51, 51, 51], dtype=torch.int64)
    crystal = CrystalGraphTensor()
 
    rcut = 5.0
    
    edges, tshifts = crystal.tensorgraph(rcut=rcut, atoms=atoms)

    verify_against_ase(crystal.pos, crystal.cell, z, edges, tshifts, cutoff=rcut)