import logging
import torch
from ase.calculators.calculator import Calculator, all_changes
from torch_geometric.data import Data

# Ensure this points to the new Unified Pipeline we just wrote!
from graph import CrystalGraphTensor, ExchangeGraphPipeline 
from model import calcforce

class DSpinGNNCalculator(Calculator):
    # Added 'local_j' so ASE knows this calculator produces custom properties
    implemented_properties = ['energy', 'forces', 'local_j']

    def __init__(self, structuremodel, exchangemodel, rcut=7.0, device='cpu', **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)
        
        self.device = torch.device(device)
        self.rcut = rcut

        # Ensure models are on the correct device and locked in evaluation mode
        self.structuremodel = structuremodel.to(self.device).eval()
        self.exchangemodel = exchangemodel.to(self.device).eval()

        # Initialize Graph Pipelines
        self.cgraph = CrystalGraphTensor()
        self.egraph = ExchangeGraphPipeline(device=self.device)

    def prepdata(self, atoms):
        """Converts ASE Atoms to a joint PyG Data object for both models."""
        
        # 1. Base Properties (Crucial: pos MUST require_grad for force calculations!)
        z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=self.device, requires_grad=True)
        cell = torch.tensor(atoms.get_cell().array, dtype=torch.float32, device=self.device)
        
        # Dummy batch tensor (required by most PyG pooling layers)
        batch = torch.zeros(len(atoms), dtype=torch.long, device=self.device)

        # 2. Structural Graph (For Energy/Forces)
        edgeidxs, edgeshifts = self.cgraph.crystalgraph(self.rcut, atoms)
        edgeidxs = edgeidxs.to(self.device)
        edgeshifts = edgeshifts.to(self.device)

        # 3. Magnetic Exchange Graph (For J values)
        # We pass xmlpath=None because this is pure simulation/inference
        cr_edges, _, cr_shifts, cr_edgedists, cr_cr_angles, cri_bonds = \
            self.egraph.process_atoms(atoms, rcut=self.rcut, xmlpath=None)

        # 4. Pack everything into a single Data object
        return Data(
            z=z, pos=pos, cell=cell, batch=batch,
            edge_index=edgeidxs, edge_shift=edgeshifts,
            cr_edge_index=cr_edges, cr_edge_shift=cr_shifts,
            cr_edge_dist=cr_edgedists, cr_cr_angles=cr_cr_angles, cr_i_bonds=cri_bonds
        )

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties
            
        super().calculate(atoms, properties, system_changes)

        # 1. Prepare joint graph
        data = self.prepdata(self.atoms)

        # ==========================================
        # 2. Forward Passes
        # ==========================================
        # Structural model maintains the PyTorch computational graph to derive forces
        energy = self.structuremodel(data)
        
        # We disable gradients for the exchange model to save memory and compute time!
        with torch.no_grad():
            exchange = self.exchangemodel(data)

        # 3. Compute Forces via Autograd
        forces = calcforce(energy, data.pos)

        # ==========================================
        # 4. Store Results for ASE
        # ==========================================
        self.results['energy'] = energy.detach().cpu().item()
        self.results['forces'] = forces.detach().cpu().numpy()

        # 5. Map Edge Exchange (J) to Atomic Nodes for OVITO Visualization
        local_j_field = torch.zeros(len(self.atoms), dtype=torch.float32, device=self.device)
        
        if exchange.numel() > 0:
            # Map J values specifically to the Chromium source atoms
            local_j_field.scatter_add_(0, data.cr_edge_index[0], exchange.view(-1).detach())

        local_j_numpy = local_j_field.cpu().numpy()
        self.atoms.set_array("Local_J", local_j_numpy)
        self.results['local_j'] = local_j_numpy