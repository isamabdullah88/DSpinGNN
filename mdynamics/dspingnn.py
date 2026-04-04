import torch
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data

class DSpinGNNCalculator(Calculator):
    # Standard ASE properties we intend to provide
    implemented_properties = ['energy', 'forces']

    def __init__(self, model, rcut=7.0, device='cpu', **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()  # Ensure dropout/batchnorm are disabled
        self.rcut = rcut   # Must match your training cutoff!

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        # 1. Extract Positions and Cell
        pos = torch.tensor(self.atoms.get_positions(), dtype=torch.float32, device=self.device, requires_grad=True)
        z = torch.tensor(self.atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)
        cell = torch.tensor(self.atoms.get_cell()[:], dtype=torch.float32, device=self.device)

        # 2. Dynamically Build the Graph for this exact timestep (Handles PBCs!)
        # i: source, j: target, S: periodic shift vectors
        i, j, S = neighbor_list('ijS', self.atoms, self.rcut)
        
        edge_index = torch.tensor(np.vstack((i, j)), dtype=torch.long, device=self.device)
        edge_shift = torch.tensor(S, dtype=torch.float32, device=self.device)

        # 3. Pack into PyTorch Geometric Batch
        batch_idx = torch.zeros(len(self.atoms), dtype=torch.long, device=self.device)
        data = Data(pos=pos, z=z, cell=cell, edge_index=edge_index, edge_shift=edge_shift, batch=batch_idx)

        # 4. Model Forward Pass (No kcal/mol conversion needed!)
        energy = self.model(data)

        # 5. Compute Forces via Autograd
        forces = -torch.autograd.grad(
            outputs=energy, 
            inputs=data.pos, 
            grad_outputs=torch.ones_like(energy),
            create_graph=False,
            retain_graph=False
        )[0]

        # 6. Store results for ASE
        self.results['energy'] = energy.detach().cpu().item()
        self.results['forces'] = forces.detach().cpu().numpy()