import torch

class CrystalGraphTensor:
    def __init__(self):
        pass

    def crystalgraph(self, rcut, atoms):
        """
        Builds a graph with periodic boundary conditions from scratch including ghost atoms using PyTorch tensors
        """
        N = atoms.get_global_number_of_atoms()
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        cell = torch.tensor(atoms.get_cell(), dtype=torch.float32)
        
        mx, my = torch.meshgrid(torch.tensor([-1, 0, 1], dtype=torch.int64), torch.tensor([-1, 0, 1], dtype=torch.int64), indexing='ij')
        mz = torch.zeros_like(mx)

        stack = torch.stack((mx.flatten(), my.flatten(), mz.flatten()), axis=1)
        
        translations = torch.mm(stack.to(torch.float32), cell)

        ghostpos = pos.unsqueeze(0) + translations.unsqueeze(1)
        ghostpos = ghostpos.view(-1, 3)

        dist = torch.cdist(ghostpos, pos)

        mask = (dist > 1e-4) & (dist <= rcut)

        ghostidxs, atomidxs = torch.where(mask)

        srcidxs = atomidxs
        dstidxs = ghostidxs % N
        cellidxs = ghostidxs // N

        homecell = (stack[cellidxs] == 0).all(dim=1)
        selfloop = (srcidxs == dstidxs) & homecell

        # Invert to keep only valid edges
        vedges = ~selfloop

        # Apply the filter
        srcidxs = srcidxs[vedges]
        dstidxs = dstidxs[vedges]
        cellidxs = cellidxs[vedges]

        shifts = stack[cellidxs]

        edges = torch.stack((srcidxs, dstidxs), axis=1)

        return edges.t(), shifts
    


if __name__ == "__main__":
    from .CrI3 import CrI3
    atoms = CrI3().batoms
    z = torch.tensor([24, 24, 51, 51, 51, 51, 51, 51], dtype=torch.int64)
    crystal = CrystalGraphTensor()
 
    rcut = 5.0
    
    edges, tshifts = crystal.crystalgraph(rcut=rcut, atoms=atoms)

    # verify_against_ase(crystal.pos, crystal.cell, z, edges, tshifts, cutoff=rcut)