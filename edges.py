import numpy as np
import matplotlib.pyplot as plt
import torch

a = 6.91
c = 19.26
dz = 1.62

veca = torch.tensor([a, 0, 0], dtype=torch.float32)
vecb = torch.tensor([-np.cos(np.pi/3)*a, np.sin(np.pi/3)*a, 0], dtype=torch.float32)
vecc = torch.tensor([0, 0, c], dtype=torch.float32)


cr1 = 1/3*veca + 2/3*vecb
cr2 = 2/3*veca + 1/3*vecb

It1 = 0.318*veca + 0.0*vecb + 0.60*vecc
It2 = 0.0*veca + 0.318*vecb + 0.60*vecc
It3 = 0.682*veca + 0.682*vecb + 0.60*vecc

Ib1 = 0.682*veca + 0.0*vecb + 0.4*vecc
Ib2 = 0.0*veca + 0.682*vecb + 0.4*vecc
Ib3 = 0.318*veca + 0.318*vecb + 0.4*vecc

def translate_cell(n1, n2):
    cr1p = cr1 + n1*veca + n2*vecb + 0*vecc
    cr2p = cr2 + n1*veca + n2*vecb + 0*vecc

    It1p = It1 + n1*veca + n2*vecb + 0*vecc
    It2p = It2 + n1*veca + n2*vecb + 0*vecc
    It3p = It3 + n1*veca + n2*vecb + 0*vecc
    Ib1p = Ib1 + n1*veca + n2*vecb + 0*vecc
    Ib2p = Ib2 + n1*veca + n2*vecb + 0*vecc
    Ib3p = Ib3 + n1*veca + n2*vecb + 0*vecc

    return cr1p, cr2p, It1p, It2p, It3p, Ib1p, Ib2p, Ib3p

def dist(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)





def bgraph(z, pos, rcut = 5.0):
    """
    Builds a graph with periodic boundary conditions from scratch including ghost atoms
    """
    graphdict = {}

    edges = []
    tshifts = []

    for i, p in enumerate(pos):
        print(p)

        for j, q in enumerate(pos):
            
            if i == j:
                continue

            d = dist(p, q)
            # print('dist: ', d)

            if d > 0 and d <= rcut:
                edges.append([i, j])
                edges.append([j, i]) # Undirected/symmetric edge
                tshifts.append([0, 0, 0])
                tshifts.append([0, 0, 0]) # Symmetric edge has same shift

        for nx, ny in [(1, 0), (-1, 0), (0, 1),  (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)]:
            cr1p, cr2p, It1p, It2p, It3p, Ib1p, Ib2p, Ib3p = translate_cell(nx, ny)

            for k, q in enumerate([cr1p, cr2p, It1p, It2p, It3p, Ib1p, Ib2p, Ib3p]):
                d = dist(p, q)
                # print('dist: ', d)

                if d > 0 and d <= rcut:
                    edges.append([i, k])
                    tshifts.append([nx, ny, 0])
                    edges.append([k, i])
                    tshifts.append([-nx, -ny, 0])

    print('edges: ', edges)
    print('tshifts: ', tshifts)
    return edges, tshifts


N = 8
z = torch.tensor(np.array([24, 24, 53, 53, 53, 53, 53, 53]))
# print(z)

pos = torch.tensor(np.array([cr1, cr2, It1, It2, It3, Ib1, Ib2, Ib3]))
print('pos: ', pos.shape)

cell = torch.stack([veca, vecb, vecc])
print('cell: ', cell.shape)
print('cell: ', cell)


# batch = [0]*8 + [1]*8 + [2]*8 + [3]*8
# print(batch)

# rcut = 6.0
# edges, tshifts = bgraph(z, pos, rcut)

# from testedges import verify_against_ase

# verify_against_ase(pos, cell, z, torch.tensor(edges).t().contiguous(), torch.tensor(tshifts), cutoff=rcut)

C = 9
mx, my = torch.meshgrid(torch.tensor([-1.0, 0, 1.0], dtype=torch.float32), torch.tensor([-1.0, 0, 1.0], dtype=torch.float32))
# print(mx.flatten().shape)
# print(my.shape)

# print('mx: ', mx.flatten())
# print('my: ', my.flatten())

stack = torch.stack((mx.flatten(), my.flatten(), torch.zeros(9)), axis=1)
print('stack: ', stack.shape)
# print('stack: ', stack)

translations = torch.mm(stack, cell)

print('translations: ', translations.shape)
# print('translations: ', translations)

ghostpos = pos.unsqueeze(0) + translations.unsqueeze(1)
ghostpos = ghostpos.view(-1, 3)

print('ghostpos: ', ghostpos.shape)

dist = torch.cdist(ghostpos, pos)

print('dist: ', dist.shape)
# print('dist: ', dist)

rcut = 5.0
mask = (dist > 1e-4) & (dist <= rcut)
print('mask: ', mask.shape)

ghostidxs, atomidxs = torch.where(mask)
print('atomidxs: ', atomidxs.shape)
print('ghostidxs: ', ghostidxs.shape)
print('atomidxs: ', atomidxs)
print('ghostidxs: ', ghostidxs)

srcidxs = ghostidxs
dstidxs = atomidxs % N
cellidxs = ghostidxs // C

edges = torch.stack((srcidxs, dstidxs), axis=1)
shifts = stack[cellidxs]

print('edges: ', edges.shape)
print('shifts: ', shifts.shape)