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





def loopgraph(z, pos, rcut = 5.0):
    """
    Builds a graph with periodic boundary conditions from scratch including ghost atoms
    """

    edges = []
    tshifts = []

    for i, p in enumerate(pos):
        print(p)

        for j, q in enumerate(pos):
            
            if i == j:
                continue

            d = dist(p, q)

            if d > 0 and d <= rcut:
                edges.append([i, j])
                edges.append([j, i]) # Undirected/symmetric edge
                tshifts.append([0, 0, 0])
                tshifts.append([0, 0, 0]) # Symmetric edge has same shift

        for nx, ny in [(1, 0), (-1, 0), (0, 1),  (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)]:
            cr1p, cr2p, It1p, It2p, It3p, Ib1p, Ib2p, Ib3p = translate_cell(nx, ny)

            for k, q in enumerate([cr1p, cr2p, It1p, It2p, It3p, Ib1p, Ib2p, Ib3p]):
                d = dist(p, q)

                if d > 0 and d <= rcut:
                    edges.append([i, k])
                    tshifts.append([nx, ny, 0])
                    edges.append([k, i])
                    tshifts.append([-nx, -ny, 0])

    return edges, tshifts

# rcut = 6.0
# edges, tshifts = bgraph(z, pos, rcut)

# from testedges import verify_against_ase

# verify_against_ase(pos, cell, z, torch.tensor(edges).t().contiguous(), torch.tensor(tshifts), cutoff=rcut)

def tensorgraph(pos, cell, rcut=5.0):
    """
    Builds a graph with periodic boundary conditions from scratch including ghost atoms using PyTorch tensors
    """
    N = 8
    
    mx, my = torch.meshgrid(torch.tensor([-1, 0, 1], dtype=torch.int64), torch.tensor([-1, 0, 1], dtype=torch.int64), indexing='ij')
    mz = torch.zeros_like(mx)
    print('mx: ', mx.dtype, mx.shape)
    print('my: ', my.dtype, my.shape)
    print('mz: ', mz.dtype, mz.shape)

    stack = torch.stack((mx.flatten(), my.flatten(), mz.flatten()), axis=1)
    print('stack: ', stack.dtype, stack.shape)
    print('stack: ', [tuple(row.tolist()) for row in stack])
    
    translations = torch.mm(stack.to(torch.float32), cell)
    print('\npositions: \n', [tuple(row.tolist()) for row in pos])
    print('\ntranslations: \n', [tuple(row.tolist()) for row in translations])

    ghostpos = pos.unsqueeze(0) + translations.unsqueeze(1)
    ghostpos = ghostpos.view(-1, 3)
    print('\nghostpos: \n', [tuple(row.tolist()) for row in ghostpos])

    dist = torch.cdist(ghostpos, pos)
    print('\ndist: ', dist.dtype, dist.shape)

    mask = (dist > 1e-4) & (dist <= rcut)

    ghostidxs, atomidxs = torch.where(mask)
    print('\nindex matrix: \n', [(i, j) for i, j in zip(ghostidxs.tolist(), atomidxs.tolist())])

    srcidxs = atomidxs
    dstidxs = ghostidxs % N
    cellidxs = ghostidxs // N
    print('\nsrcidxs: ', srcidxs)
    print('\ndstidxs: ', dstidxs)
    print('\ncellidxs: ', cellidxs)

    edges = torch.stack((srcidxs, dstidxs), axis=1)
    print('\nedges: ', edges.dtype, edges.shape)
    print('\nedges: ', [tuple(row.tolist()) for row in edges])
    shifts = stack[cellidxs]
    print('\nshifts: ', [tuple(row.tolist()) for row in shifts])
    # exit()

    return edges.t(), shifts


from testedges import verify_against_ase

z = torch.tensor([24, 24, 51, 51, 51, 51, 51, 51], dtype=torch.int64)
pos = torch.tensor(np.array([cr1, cr2, It1, It2, It3, Ib1, Ib2, Ib3]), dtype=torch.float32)
cell = torch.stack([veca, vecb, vecc])
rcut = 5.0

edges, tshifts = tensorgraph(pos, cell, rcut=rcut)
# print('edges: ', [tuple(edge) for edge in edges.tolist()])
# print('tshifts: ', [tuple(shift) for shift in tshifts.tolist()])

verify_against_ase(pos, cell, z, edges, tshifts, cutoff=rcut)