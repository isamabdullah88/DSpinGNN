import xml.etree.ElementTree as ET
import logging

import torch
import numpy as np
from ase.neighborlist import neighbor_list
from .geometry import GeometryExtractor
from .tb2j import TB2JParser


# ==========================================
# 3. Unified Graph Pipeline
# ==========================================
class ExchangeGraphPipeline:
    """Orchestrates the conversion of ASE Atoms to PyTorch Geometric graphs."""
    def __init__(self, device='cpu', magnetic_z=24, ligand_symbol='I'):
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.magnetic_z = magnetic_z
        self.geo_extractor = GeometryExtractor(device=device, ligand_symbol=ligand_symbol)

    def process_atoms(self, atoms, rcut=4.5, xmlpath=None):
        """
        The unified entry point.
        If xmlpath is provided -> Training Mode (loads J targets).
        If xmlpath is None -> Simulation Mode (J targets are 0).
        """
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=self.device)
        cell = torch.tensor(atoms.get_cell().array, dtype=torch.float32, device=self.device)
        symbols = atoms.get_chemical_symbols()
        z_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)

        # 1. Pre-compute ligand supercell
        all_ligand_pos = self.geo_extractor.build_ligand_supercell(positions, symbols, cell)

        # 2. Determine Edges
        if xmlpath:
            edges, shifts, exchangejs = self._get_edges_from_tb2j(xmlpath)
        else:
            edges, shifts, exchangejs = self._get_edges_from_ase(atoms, z_numbers, rcut)

        # 3. Unified Physics Extraction Loop
        cr_edges, valid_shifts, valid_js = [], [], []
        distances, angles, bonds = [], [], []

        for edge, shift, jval in zip(edges.t(), shifts, exchangejs):
            i_idx, j_idx = edge.tolist()
            
            ipos = positions[i_idx]
            fpos = positions[j_idx] + (shift @ cell)
            
            dist = torch.norm(ipos - fpos)
            if dist > rcut:
                continue

            # Extract features
            avg_cos_theta, cri_bonds = self.geo_extractor.get_angles_and_bonds(all_ligand_pos, ipos, fpos)

            # Store valid data
            cr_edges.append([i_idx, j_idx])
            valid_shifts.append(shift.tolist())
            valid_js.append(jval.item())
            distances.append(dist.item())
            angles.append(avg_cos_theta)
            bonds.append(cri_bonds)

        # 4. Convert all lists directly to final PyTorch tensors
        cr_edge_index = torch.tensor(cr_edges, dtype=torch.long, device=self.device).t().contiguous()
        cr_edge_shift = torch.tensor(valid_shifts, dtype=torch.long, device=self.device)
        exchangejs = torch.tensor(valid_js, dtype=torch.float32, device=self.device).view(-1, 1)
        cr_distances = torch.tensor(distances, dtype=torch.float32, device=self.device)
        cr_cr_angles = torch.tensor(angles, dtype=torch.float32, device=self.device).view(-1, 1)
        cri_bonds_tensor = torch.tensor(bonds, dtype=torch.float32, device=self.device).view(-1, 4)

        return cr_edge_index, exchangejs, cr_edge_shift, cr_distances, cr_cr_angles, cri_bonds_tensor

    def _get_edges_from_tb2j(self, xmlpath):
        tb2j_dict = TB2JParser.parse(xmlpath)
        edges, shifts, js = [], [], []
        for (i, j, Rx, Ry, Rz), jval in tb2j_dict.items():
            edges.append([i, j])
            shifts.append([Rx, Ry, Rz])
            js.append(jval)
        
        return (torch.tensor(edges, device=self.device).t(), 
                torch.tensor(shifts, dtype=torch.float32, device=self.device), 
                torch.tensor(js, device=self.device))

    def _get_edges_from_ase(self, atoms, z_numbers, rcut):
        i, j, S = neighbor_list('ijS', atoms, rcut)
        
        edge_index = torch.tensor(np.vstack((i, j)), dtype=torch.long, device=self.device)
        edge_shift = torch.tensor(S, dtype=torch.float32, device=self.device)

        # Mask for magnetic atoms (e.g. Z=24 for Cr)
        is_mag = (z_numbers == self.magnetic_z)
        src, dst = edge_index
        mag_mask = is_mag[src] & is_mag[dst]

        filtered_edges = edge_index[:, mag_mask]
        filtered_shifts = edge_shift[mag_mask]
        dummy_js = torch.zeros(filtered_edges.shape[1], device=self.device)

        return filtered_edges, filtered_shifts, dummy_js