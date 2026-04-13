import numpy as np
from ase.visualize import view

class StrainEngineer:
    """Class to handle macroscopic geometry deformations of ASE atoms."""
    
    def __init__(self, amplitude: float, strain_type: str = "biaxial", 
                 qx: int = None, qy: int = None, qc: int = None):
        self.amplitude = amplitude
        self.strain_type = strain_type.lower()
        self.qx = qx
        self.qy = qy
        self.qc = qc

    def apply_ripple(self, atoms, logger=None):
        """Applies an in-plane density wave based on the configured strain type."""
        positions = atoms.get_positions()
        
        # 1. Safely extract exact cell boundaries AFTER the FIRE relaxation
        Lx = atoms.cell[0][0]
        Ly = atoms.cell[1][1]
        
        # 2. Calculate Wavelengths based on number of quarter-waves (q)
        # Formula: Wavelength = 4 * Length / q
        # Default to 4 (exactly 1 full wave) if None is provided.
        wx = (4.0 * Lx / self.qx) if self.qx is not None else Lx
        wy = (4.0 * Ly / self.qy) if self.qy is not None else Ly
        
        # For circular waves, the boundary is the shortest distance from center to edge
        max_radius = min(Lx, Ly) / 2.0
        wc = (4.0 * max_radius / self.qc) if self.qc is not None else max_radius
        
        if logger:
            logger.info(f"Applying {self.strain_type} wave (Amp: {self.amplitude} Å)")
            logger.info(f"Quarter-waves -> Qx: {self.qx}, Qy: {self.qy}, Qc: {self.qc}")
            logger.info(f"Calculated Wavelengths -> Wx: {wx:.2f}, Wy: {wy:.2f}, Wc: {wc:.2f}")

        if self.strain_type == "uniaxial":
            positions[:, 0] += self.amplitude * np.sin(2 * np.pi * positions[:, 0] / wx)
            
        elif self.strain_type == "biaxial":
            positions[:, 0] += self.amplitude * np.sin(2 * np.pi * positions[:, 0] / wx)
            positions[:, 1] += self.amplitude * np.sin(2 * np.pi * positions[:, 1] / wy)
            
        elif self.strain_type in ["circular", "circular_shear"]:
            xc, yc = Lx / 2.0, Ly / 2.0
            
            dx = positions[:, 0] - xc
            dy = positions[:, 1] - yc
            r = np.sqrt(dx**2 + dy**2)
            r[r == 0] = 1e-10 
            
            if self.strain_type == "circular":
                radial_disp = self.amplitude * np.sin(2 * np.pi * r / wc)
                positions[:, 0] += radial_disp * (dx / r)
                positions[:, 1] += radial_disp * (dy / r)
                
            elif self.strain_type == "circular_shear":
                shear_disp = self.amplitude * np.sin(2 * np.pi * r / wc)
                positions[:, 0] += shear_disp * (-dy / r)
                positions[:, 1] += shear_disp * (dx / r)
                
        # Update geometry
        atoms.set_positions(positions)

        view(atoms)  # Visualize the strained structure
        
        return atoms