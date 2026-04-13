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
        self.model.eval()  
        self.rcut = rcut   

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        # 1. Extract Positions and Cell
        pos = torch.tensor(self.atoms.get_positions(), dtype=torch.float32, device=self.device, requires_grad=True)
        z = torch.tensor(self.atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)
        cell = torch.tensor(self.atoms.get_cell()[:], dtype=torch.float32, device=self.device)

        # 2. Dynamically Build the Universal Graph
        i, j, S = neighbor_list('ijS', self.atoms, self.rcut)
        
        edge_index = torch.tensor(np.vstack((i, j)), dtype=torch.long, device=self.device)
        edge_shift = torch.tensor(S, dtype=torch.float32, device=self.device)

        # =================================================================
        # THE FIX: Isolate the Cr-Cr Subgraph for the Exchange Block
        # =================================================================
        # Create a mask where BOTH the source and destination atoms are Chromium (Z=24)
        is_cr = (z == 24)
        src, dst = edge_index
        cr_edge_mask = is_cr[src] & is_cr[dst]
        
        # Filter the edges and shifts to only include Cr-Cr bonds
        cr_edge_index = edge_index[:, cr_edge_mask]
        cr_edge_shift = edge_shift[cr_edge_mask]

        # 3. Pack into PyTorch Geometric Batch
        batch_idx = torch.zeros(len(self.atoms), dtype=torch.long, device=self.device)
        data = Data(
            pos=pos, 
            z=z, 
            cell=cell, 
            edge_index=edge_index, 
            edge_shift=edge_shift, 
            cr_edge_index=cr_edge_index, # <--- Attached for the ExchangeBlock
            cr_edge_shift=cr_edge_shift, # <--- Attached for the ExchangeBlock
            batch=batch_idx
        )

        # 4. Model Forward Pass
        energy, exchange = self.model(data)
        # print('exchange: ', exchange.detach().cpu().numpy())  # Debug print to verify exchange values

        # 5. Compute Forces via Autograd
        forces = -torch.autograd.grad(
            outputs=energy, 
            inputs=data.pos, 
            grad_outputs=torch.ones_like(energy),
            create_graph=False,
            retain_graph=False
        )[0]

        # 6. Store Standard ASE Results
        self.results['energy'] = energy.detach().cpu().item()
        self.results['forces'] = forces.detach().cpu().numpy()

        # 7. Map Edge Exchange (J) to Atomic Nodes for OVITO
        local_j_field = torch.zeros(len(self.atoms), dtype=torch.float32, device=self.device)

        # IMPORTANT: We must use cr_edge_index[0] here so the J values are 
        # mapped to the Chromium atoms, not mistakenly mapped to Iodine atoms!
        if exchange.numel() > 0:
            local_j_field.scatter_add_(0, cr_edge_index[0], exchange.view(-1).detach())

        # Attach the Local_J array for the XYZ writer
        self.atoms.set_array("Local_J", local_j_field.cpu().numpy())
        self.results['local_j'] = local_j_field.cpu().numpy()