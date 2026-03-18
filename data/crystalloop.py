import numpy as np
import torch

class CrystalGraphLoop:
    def __init__(self, a = 6.91, c = 19.26, dz = 1.62):
        self.a = a
        self.c = c
        self.dz = dz

        # Lattice vectors for hexagonal cell
        self.veca = torch.tensor([a, 0, 0], dtype=torch.float32)
        self.vecb = torch.tensor([-np.cos(np.pi/3)*a, np.sin(np.pi/3)*a, 0], dtype=torch.float32)
        self.vecc = torch.tensor([0, 0, c], dtype=torch.float32)

        # Atomic positions in the unit cell
        self.cr1 = 1/3*self.veca + 2/3*self.vecb
        self.cr2 = 2/3*self.veca + 1/3*self.vecb

        self.It1 = 0.318*self.veca + 0.0*self.vecb + 0.60*self.vecc
        self.It2 = 0.0*self.veca + 0.318*self.vecb + 0.60*self.vecc
        self.It3 = 0.682*self.veca + 0.682*self.vecb + 0.60*self.vecc

        self.Ib1 = 0.682*self.veca + 0.0*self.vecb + 0.4*self.vecc
        self.Ib2 = 0.0*self.veca + 0.682*self.vecb + 0.4*self.vecc
        self.Ib3 = 0.318*self.veca + 0.318*self.vecb + 0.4*self.vecc

        self.pos = torch.tensor(np.array([self.cr1, self.cr2, self.It1, self.It2, self.It3, self.Ib1, self.Ib2, self.Ib3]), dtype=torch.float32)
        self.cell = torch.stack((self.veca, self.vecb, self.vecc), dim=0)


    def translate_cell(self, n1, n2):
        cr1p = self.cr1 + n1*self.veca + n2*self.vecb + 0*self.vecc
        cr2p = self.cr2 + n1*self.veca + n2*self.vecb + 0*self.vecc

        It1p = self.It1 + n1*self.veca + n2*self.vecb + 0*self.vecc
        It2p = self.It2 + n1*self.veca + n2*self.vecb + 0*self.vecc
        It3p = self.It3 + n1*self.veca + n2*self.vecb + 0*self.vecc
        Ib1p = self.Ib1 + n1*self.veca + n2*self.vecb + 0*self.vecc
        Ib2p = self.Ib2 + n1*self.veca + n2*self.vecb + 0*self.vecc
        Ib3p = self.Ib3 + n1*self.veca + n2*self.vecb + 0*self.vecc

        return cr1p, cr2p, It1p, It2p, It3p, Ib1p, Ib2p, Ib3p

    def dist(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2)

    def graph(self, rcut = 5.0):
        """
        Builds a graph with periodic boundary conditions from scratch including ghost atoms
        """
        
        edges = []
        tshifts = []

        for i, p in enumerate(self.pos):
            print(p)

            for j, q in enumerate(self.pos):
                
                if i == j:
                    continue

                d = self.dist(p, q)

                if d > 1e-4 and d <= rcut:
                    edges.append([i, j])
                    edges.append([j, i]) # Undirected/symmetric edge
                    tshifts.append([0, 0, 0])
                    tshifts.append([0, 0, 0]) # Symmetric edge has same shift

            for nx, ny in [(1, 0), (-1, 0), (0, 1),  (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)]:
                cr1p, cr2p, It1p, It2p, It3p, Ib1p, Ib2p, Ib3p = self.translate_cell(nx, ny)

                for k, q in enumerate([cr1p, cr2p, It1p, It2p, It3p, Ib1p, Ib2p, Ib3p]):
                    d = self.dist(p, q)

                    if d > 1e-4 and d <= rcut:
                        edges.append([i, k])
                        tshifts.append([nx, ny, 0])
                        edges.append([k, i])
                        tshifts.append([-nx, -ny, 0])

        return edges, tshifts


if __name__ == "__main__":
    from data.testedges import verify_against_ase

    z = torch.tensor([24, 24, 51, 51, 51, 51, 51, 51], dtype=torch.int64)
    crystal = CrystalGraphLoop()

    rcut = 5.0
    
    edges, tshifts = crystal.graph(rcut=rcut)

    verify_against_ase(crystal.pos, crystal.cell, z, edges, tshifts, cutoff=rcut)