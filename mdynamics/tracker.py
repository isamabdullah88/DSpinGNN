import numpy as np
from pathlib import Path

class MaxForceTracker:
    """Calculates max forces and checks for energy conservation/explosions."""
    def __init__(self, atoms, dyn, xyz_path, logger=None):
        self.atoms = atoms
        self.dyn = dyn
        self.logger = logger
        
        # Parse the original XYZ path and construct the CSV path
        original_path = Path(xyz_path)
        directory = original_path.parent
        base_name = original_path.stem
        
        self.log_file = directory / f"{base_name}_forces.csv"
        
        # Initialize the file with headers
        with open(self.log_file, "w") as f:
            f.write("Step,Max_Force_eV_A,Total_Energy_eV\n")

    def __call__(self):
        step = self.dyn.get_number_of_steps()
        
        # Calculate forces
        forces = self.atoms.get_forces()
        max_force = np.max(np.linalg.norm(forces, axis=1))
        
        # Calculate energy
        etot = self.atoms.get_potential_energy() + self.atoms.get_kinetic_energy()
        
        # Append to CSV
        with open(self.log_file, "a") as f:
            f.write(f"{step},{max_force:.4f},{etot:.4f}\n")
            
        # Terminal warning for out-of-distribution force spikes (> 1.5 eV/Å)
        if max_force > 1.5:
            msg = f"⚠️ WARNING: Force spike at step {step}! Max Force = {max_force:.2f} eV/Å"
            if self.logger:
                self.logger.warning(msg)
            else:
                print(msg)