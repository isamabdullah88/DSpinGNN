import numpy as np
from ase.visualize import view

class StrainEngineer:
    """Class to handle macroscopic geometry deformations of ASE atoms."""
    
    def __init__(self, amplitude: float, strain_type: str = "biaxial"):
        self.amplitude = amplitude
        self.strain_type = strain_type.lower()

    def apply_ripple(self, atoms, logger=None):
        """Applies an in-plane density wave based on the configured strain type."""
        if logger:
            logger.info(f"Applying {self.strain_type} in-plane density wave (Amp: {self.amplitude} Å)...")
            
        positions = atoms.get_positions()
        
        # Get box lengths for commensurability
        Lx = atoms.cell[0][0]
        Ly = atoms.cell[1][1]
        
        # 1. Always apply the X-wave
        positions[:, 0] += self.amplitude * np.sin(2 * np.pi * positions[:, 0] / Lx)
        
        # 2. Only apply the Y-wave if strictly biaxial
        if self.strain_type == "biaxial":
            positions[:, 1] += self.amplitude * np.sin(2 * np.pi * positions[:, 1] / Ly)
            
        # Update geometry
        atoms.set_positions(positions)
        
        # NOTE: Uncomment the line below if you want the GUI to pop up 
        # before the simulation starts to visually verify the wave!
        # view(atoms)
        
        return atoms