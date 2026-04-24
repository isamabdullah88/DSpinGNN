from pathlib import Path
import torch
import numpy as np
from torch_geometric.data import Data
from graph import EspressoHubbard, CrystalGraphTensor, ExchangeGraph

import logging

class SampleProcessor:
    """Handles parsing DFT and TB2J outputs for a single crystal directory."""
    def __init__(self, rcut, force_thresh=5.0, exchange_range=(-1.5, 3.0)):
        self.rcut = rcut
        self.fthresh = force_thresh
        self.j_range = exchange_range
        self.logger = logging.getLogger(__name__)

        # Initialize graph builders
        self.hubbard = EspressoHubbard()
        self.cgraph = CrystalGraphTensor()
        self.egraph = ExchangeGraph() # Using our newly unified class!

    def process(self, sample_dir: Path):
        """Parses a directory and returns a PyG Data object or None if invalid."""
        pwipath = sample_dir / "espresso.pwi"
        pwopath = sample_dir / "espresso.pwo"
        tb2jpath = sample_dir / "tmp" / "TB2J_results" / "Multibinit" / "exchange.xml"

        if not pwipath.exists() or not pwopath.exists():
            self.logger.warning(f"[{sample_dir.name}] Missing QE input/output files. Skipping.")
            return None

        # 1. Parse Quantum Espresso Output
        try:
            atomsout = self.hubbard.parse(str(pwipath), str(pwopath))
        except Exception as e:
            self.logger.warning(f"[{sample_dir.name}] Failed to parse QE files: {e}. Skipping.")
            return None

        forces = atomsout.get_forces()
        
        # 2. Filter by Force limits
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > self.fthresh:
            self.logger.warning(f"[{sample_dir.name}] Large forces detected ({max_force:.4f} eV/Å). Skipping.")
            return None

        # 3. Build Graphs
        edgeidxs, edgeshifts = self.cgraph.crystalgraph(self.rcut, atomsout)

        if not tb2jpath.exists():
            self.logger.warning(f"[{sample_dir.name}] Missing TB2J exchange.xml. Skipping.")
            return None
            
        # CRITICAL UPDATE: Using the new unified pipeline method
        credges, exchangejs, eshifts, distances, cosangles, bonds, angles = \
            self.egraph.process_atoms(atomsout, rcut=self.rcut, xmlpath=str(tb2jpath))

        # 4. Filter by Exchange limits (Your -1.5 to 3.0 meV scope)
        if not ((exchangejs > self.j_range[0]) & (exchangejs < self.j_range[1])).all():
            max_j = torch.max(torch.abs(exchangejs)).item()
            self.logger.warning(f"[{sample_dir.name}] Exchange out of bounds (Max |J|: {max_j:.4f} meV). Skipping.")
            return None

        # 5. Assemble and Return PyG Data
        return Data(
            z=torch.tensor(atomsout.get_atomic_numbers(), dtype=torch.long),
            pos=torch.tensor(atomsout.get_positions(), dtype=torch.float32),
            cell=torch.tensor(atomsout.get_cell().array, dtype=torch.float32),
            y_energy=torch.tensor([atomsout.get_potential_energy()], dtype=torch.float32),
            y_forces=torch.tensor(forces, dtype=torch.float32),
            edge_index=edgeidxs,
            edge_shift=edgeshifts,
            cr_edge_index=credges,
            cr_edge_shift=eshifts,
            cr_edge_dist=distances,
            cr_cr_angles=cosangles,
            cr_i_bonds=bonds,
            y_exchange=exchangejs
        )