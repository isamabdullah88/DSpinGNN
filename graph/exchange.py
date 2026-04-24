import logging

import torch
import numpy as np
from ase.neighborlist import neighbor_list
from .geometry import GeometryExtractor
from .tb2j import TB2JParser


class ExchangeGraph:
    """Orchestrates the conversion of ASE Atoms to PyTorch Geometric graphs."""
    def __init__(self, device='cpu', magnetic_z=24, lsymbol='I'):
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.magnetic_z = magnetic_z
        self.geometry = GeometryExtractor(device=device, lsymbol=lsymbol)

    def process_atoms(self, atoms, rcut=4.5, xmlpath=None):
        """
        The unified entry point.
        If xmlpath is provided -> Training Mode (loads J targets).
        If xmlpath is None -> Simulation Mode (J targets are 0).
        """
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=self.device)
        cell = torch.tensor(atoms.get_cell().array, dtype=torch.float32, device=self.device)
        symbols = atoms.get_chemical_symbols()
        znums = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)

        # 1. Pre-compute ligand supercell
        lpositions = self.geometry.ligandscell(positions, symbols, cell)

        # 2. Determine Edges
        if xmlpath:
            edges, shifts, exchangejs = self._edgestb2j(xmlpath)
        else:
            edges, shifts, exchangejs = self._edges(atoms, znums, rcut)

        # 3. Unified Physics Extraction Loop
        credges, vshifts, vexchangejs = [], [], []
        distances, cosangles, bonds, angles = [], [], [], []

        for edge, shift, jval in zip(edges.t(), shifts, exchangejs):
            idxi, idxj = edge.tolist()
            
            ipos = positions[idxi]
            fpos = positions[idxj] + (shift @ cell)
            
            dist = torch.norm(ipos - fpos)
            if dist > rcut:
                continue

            # Extract features
            avgcosangle, cribonds, avgangle = self.geometry.calc_bondsangles(lpositions, ipos, fpos)

            # Store valid data
            credges.append([idxi, idxj])
            vshifts.append(shift.tolist())
            vexchangejs.append(jval.item())
            distances.append(dist.item())
            cosangles.append(avgcosangle)
            bonds.append(cribonds)
            angles.append(avgangle)

        # 4. Convert all lists directly to final PyTorch tensors
        credges = torch.tensor(credges, dtype=torch.long, device=self.device).t().contiguous()
        eshifts = torch.tensor(vshifts, dtype=torch.long, device=self.device)
        exchangejs = torch.tensor(vexchangejs, dtype=torch.float32, device=self.device).view(-1, 1)
        distances = torch.tensor(distances, dtype=torch.float32, device=self.device)
        cosangles = torch.tensor(cosangles, dtype=torch.float32, device=self.device).view(-1, 1)
        bonds = torch.tensor(bonds, dtype=torch.float32, device=self.device).view(-1, 4)
        angles = torch.tensor(angles, dtype=torch.float32, device=self.device).view(-1, 1)

        return credges, exchangejs, eshifts, distances, cosangles, bonds, angles

    def _edgestb2j(self, xmlpath):
        tb2j_dict = TB2JParser.parse(xmlpath)
        edges, shifts, js = [], [], []
        for (i, j, Rx, Ry, Rz), jval in tb2j_dict.items():
            edges.append([i, j])
            shifts.append([Rx, Ry, Rz])
            js.append(jval)
        
        return (torch.tensor(edges, device=self.device).t(), 
                torch.tensor(shifts, dtype=torch.float32, device=self.device), 
                torch.tensor(js, device=self.device))

    def _edges(self, atoms, znums, rcut):
        i, j, S = neighbor_list('ijS', atoms, rcut)
        
        edgeidxs = torch.tensor(np.vstack((i, j)), dtype=torch.long, device=self.device)
        eshifts = torch.tensor(S, dtype=torch.float32, device=self.device)

        # Mask for magnetic atoms (e.g. Z=24 for Cr)
        ismag = (znums == self.magnetic_z)
        src, dst = edgeidxs
        magmask = ismag[src] & ismag[dst]

        edges = edgeidxs[:, magmask]
        shifts = eshifts[magmask]
        exchangejs = torch.zeros(edges.shape[1], device=self.device)

        return edges, shifts, exchangejs