import torch
import numpy as np
from ase.md.langevin import Langevin
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.optimize import FIRE

from data import CrI3
from .dspingnn import DSpinGNNCalculator
from model import DSpinGNN
from train.trainutils import load_checkpoint

from .strains import StrainEngineer
from .tracker import MaxForceTracker
import os

import logging
from tqdm import tqdm


class CrI3_Simulator:
    """Modular class to handle ML-driven Molecular Dynamics for CrI3."""
    
    def __init__(self, modelpath, nx, ny, tmpK, timesteps, amplitude, strain_type, out_dir="Simulations"):
        # Physical & Simulation Parameters
        self.modelpath = modelpath
        self.nx = nx
        self.ny = ny
        self.tmpK = tmpK
        self.timesteps = timesteps
        self.amplitude = amplitude
        self.strain_type = strain_type
        
        # System Setup
        self.logger = self._setup_logger()
        self.device = self._detect_hardware()
        
        # Dynamically Parameterize the Output Filename
        os.makedirs(out_dir, exist_ok=True)
        filename = f"CrI3_MD_{self.strain_type}_{self.nx}x{self.ny}_T{self.tmpK}K_Amp{self.amplitude}A_{self.timesteps}steps.xyz"
        self.outfile = os.path.join(out_dir, filename)
        
        self.logger.info(f"Initialized Simulator. Output will be saved to: {self.outfile}")

    def _setup_logger(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
        return logging.getLogger(__name__)

    def _detect_hardware(self):
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        self.logger.info(f"Hardware detected! Routing tensors to: {device.upper()}")
        return device

    def _load_model(self):
        self.logger.info("Loading PyTorch MLIP model...")
        model = DSpinGNN()
        model, _, _, _ = load_checkpoint(model, self.modelpath, self.device)
        return model

    def build_lattice(self):
        """Generates the supercell and applies the configured strain."""
        self.logger.info(f"Generating pristine CrI3 {self.nx}x{self.ny} lattice...")
        cri3_manager = CrI3()
        patoms = cri3_manager.batoms.copy()
        
        # Build Supercell
        atoms = patoms * (self.nx, self.ny, 1)
        
        return atoms
    
    def apply_strain(self, atoms):
        """Applies the configured strain to the atoms."""
        self.logger.info(f"Applying {self.strain_type} strain with amplitude {self.amplitude} Å...")
        strain_engineer = StrainEngineer(amplitude=self.amplitude, strain_type=self.strain_type)
        strained_atoms = strain_engineer.apply_ripple(atoms, logger=self.logger)
        return strained_atoms

    def run(self):
        """Executes the main molecular dynamics loop."""
        # 1. Build and Strain the Atoms
        atoms = self.build_lattice()
        
        # 2. Attach MLIP Calculator
        model = self._load_model()
        atoms.calc = DSpinGNNCalculator(model=model, rcut=7.0, device=self.device)

        # =========================================================
        # NEW: PRE-RELAX THE LATTICE
        # =========================================================
        self.logger.info("Optimizing lattice to the MLIP's exact equilibrium...")
        opt = FIRE(atoms, logfile=None) # logfile=None keeps the terminal clean
        opt.run(fmax=0.01) # Run until the maximum force is safely below 0.01 eV/A
        self.logger.info("Optimization complete!")
        self.logger.info(f"Post-optimization lattice constants: {atoms.cell[0][0]:.3f} Å (X), {atoms.cell[1][1]:.3f} Å (Y)")
        # =========================================================
        
        # Apply the strain after optimization to ensure we start from the MLIP's ideal geometry
        atoms = self.apply_strain(atoms)

        # Pre-register the array so ASE knows it exists
        atoms.set_array("Local_J", np.zeros(len(atoms)))

        # 3. Thermalize
        self.logger.info(f"Thermalizing to {self.tmpK}K...")
        MaxwellBoltzmannDistribution(atoms, temperature_K=self.tmpK)
        Stationary(atoms)
        ZeroRotation(atoms)

        # 4. Setup Dynamics Engine
        dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=self.tmpK, friction=0.01)

        # 5. Clear old file and Setup Writer
        open(self.outfile, 'w').close()

        def write_frame():
            if 'local_j' in atoms.calc.results:
                atoms.set_array("Local_J", atoms.calc.results['local_j'])
            write(self.outfile, atoms, format='extxyz', append=True)

        dyn.attach(write_frame, interval=5)
        
        # =========================================================
        # ATTACH THE FORCE TRACKER
        # =========================================================
        self.logger.info("Initializing Maximum Force Tracker...")
        tracker = MaxForceTracker(atoms=atoms, dyn=dyn, xyz_path=self.outfile, logger=self.logger)
        dyn.attach(tracker, interval=10)

        # 6. Setup Progress Bar
        pbar_interval = 10
        pbar = tqdm(total=self.timesteps, desc=f"MD Simulation ({self.tmpK}K)", unit="step", dynamic_ncols=True)

        def update_pbar():
            pbar.update(pbar_interval)

        dyn.attach(update_pbar, interval=pbar_interval)

        # 7. Execute
        self.logger.info("Starting Simulation...")
        dyn.run(self.timesteps)
        pbar.close()
        self.logger.info(f"MD Finished! Open '{self.outfile}' in OVITO.")
        self.logger.info(f"Force logs saved to '{tracker.log_file}'.")


if __name__ == "__main__":
    
    # =========================================================
    # SIMULATION CONTROL CENTER
    # =========================================================
    
    config = {
        "modelpath": "checkpoints/Full-Structural-Model(No-exchange)-1/Epoch-3500.pt",
        "nx": 5,                     # Unit cells in X direction
        "ny": 5,                     # Unit cells in Y direction (5x1 Ribbon optimized for local testing)
        "tmpK": 20,                  # Temperature in Kelvin
        "timesteps": 10000,          # Total MD steps
        "amplitude": 0.5,          # Safe amplitude in Å (Under 5% strain limit)
        "strain_type": "uniaxial"    # "uniaxial" or "biaxial"
    }
    
    # Initialize and run
    simulator = CrI3_Simulator(**config)
    simulator.run()