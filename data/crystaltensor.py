import torch
import numpy as np

class CrystalGraphTensor:
    def __init__(self, a = 6.91, c = 19.26, dz = 1.62):
        self.a = a
        self.c = c
        self.dz = dz

        # Lattice vectors for hexagonal cell
        veca = torch.tensor([6.91, 0, 0], dtype=torch.float32)
        vecb = torch.tensor([-6.91*np.cos(np.pi/3), 6.91*np.sin(np.pi/3), 0], dtype=torch.float32)
        vecc = torch.tensor([0, 0, 19.26], dtype=torch.float32)
        self.cell = torch.stack((veca, vecb, vecc), dim=0)
        
        # Atomic positions in the unit cell
        cr1 = 1/3*veca + 2/3*vecb
        cr2 = 2/3*veca + 1/3*vecb

        It1 = 0.318*veca + 0.0*vecb + 0.60*vecc
        It2 = 0.0*veca + 0.318*vecb + 0.60*vecc
        It3 = 0.682*veca + 0.682*vecb + 0.60*vecc

        Ib1 = 0.682*veca + 0.0*vecb + 0.4*vecc
        Ib2 = 0.0*veca + 0.682*vecb + 0.4*vecc
        Ib3 = 0.318*veca + 0.318*vecb + 0.4*vecc

        self.pos = torch.tensor(np.array([cr1, cr2, It1, It2, It3, Ib1, Ib2, Ib3]), dtype=torch.float32)


    def tensorgraph(self, rcut):
        """
        Builds a graph with periodic boundary conditions from scratch including ghost atoms using PyTorch tensors
        """
        N = 8
        
        mx, my = torch.meshgrid(torch.tensor([-1, 0, 1], dtype=torch.int64), torch.tensor([-1, 0, 1], dtype=torch.int64), indexing='ij')
        mz = torch.zeros_like(mx)
        # print('mx: ', mx.dtype, mx.shape)
        # print('my: ', my.dtype, my.shape)
        # print('mz: ', mz.dtype, mz.shape)

        stack = torch.stack((mx.flatten(), my.flatten(), mz.flatten()), axis=1)
        # print('stack: ', stack.dtype, stack.shape)
        # print('stack: ', [tuple(row.tolist()) for row in stack])
        
        translations = torch.mm(stack.to(torch.float32), self.cell)
        # print('\npositions: \n', [tuple(row.tolist()) for row in self.pos])
        # print('\ntranslations: \n', [tuple(row.tolist()) for row in translations])

        ghostpos = self.pos.unsqueeze(0) + translations.unsqueeze(1)
        ghostpos = ghostpos.view(-1, 3)
        # print('\nghostpos: \n', [tuple(row.tolist()) for row in ghostpos])

        dist = torch.cdist(ghostpos, self.pos)
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

        edges = torch.stack((srcidxs, dstidxs), axis=1)
        # print('\nedges: ', edges.dtype, edges.shape)
        # print('\nedges: ', [tuple(row.tolist()) for row in edges])
        shifts = stack[cellidxs]
        # print('\nshifts: ', [tuple(row.tolist()) for row in shifts])

        return edges.t(), shifts
    


if __name__ == "__main__":
    from data.testedges import verify_against_ase
 
    z = torch.tensor([24, 24, 51, 51, 51, 51, 51, 51], dtype=torch.int64)
    crystal = CrystalGraphTensor()
 
    rcut = 5.0
    
    edges, tshifts = crystal.tensorgraph(rcut=rcut)

    verify_against_ase(crystal.pos, crystal.cell, z, edges, tshifts, cutoff=rcut)