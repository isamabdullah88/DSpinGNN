import torch
import numpy as np
from ase.calculators.calculator import Calculator, all_changes

class NequIPCalculator(Calculator):
    # Standard ASE properties we intend to provide
    implemented_properties = ['energy', 'forces']

    def __init__(self, model, device='cpu', **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()  # Ensure dropout/batchnorm are in eval mode

    def calculate(self, atoms=None, properties=['energy'], 
                  system_changes=all_changes):
        # 1. Update the atoms object and state
        super().calculate(atoms, properties, system_changes)

        # 2. Prepare inputs from ASE Atoms object
        # Note: NequIP typically needs positions, atomic numbers, and 
        # potentially an edge list (neighbor list) if you don't compute it inside the model.
        pos = torch.tensor(self.atoms.get_positions(), 
                           dtype=torch.float32, 
                           device=self.device, 
                           requires_grad=True)
        z = torch.tensor(self.atoms.get_atomic_numbers(), 
                         dtype=torch.long, 
                         device=self.device)
        
        # Inside your calculate() method:
        num_atoms = len(self.atoms)
        batch = torch.zeros(num_atoms, dtype=torch.long, device=self.device)

        # Then call your model:
        # 3. Model Forward Pass
        energy = self.model(z, pos, batch) * 0.04336 # kcal/mol to eV conversion

        # 4. Compute Forces via Autograd
        # Forces = -d(Energy)/d(Positions)
        forces = -torch.autograd.grad(energy, pos, 
                                      grad_outputs=torch.ones_like(energy),
                                      retain_graph=False)[0]

        # 5. Store results in ASE-friendly format (numpy arrays)
        self.results['energy'] = energy.item()
        self.results['forces'] = forces.detach().cpu().numpy()