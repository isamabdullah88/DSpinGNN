import os
import logging

import torch
import numpy as np
from tqdm import tqdm

from ase.md.langevin import Langevin
from ase import units
from ase.io import write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.optimize import FIRE

# Ensure these match your project structure
from .simconfig import SimConfig
from graph import CrI3
from .dspingnn import DSpinGNNCalculator
from model import StructureGNN, ExchangeMLP
from train.trainutils import load_checkpoint
from .strains import StrainEngineer
from .tracker import MaxForceTracker

# =========================================================
# 2. Molecular Dynamics Simulator
# =========================================================
class CrI3_Simulator:
    def __init__(self, config: SimConfig):
        self.config = config
        
        # Automatically connects to your root logger from logger.py!
        self.logger = logging.getLogger(__name__)
        self.device = self._detect_hardware()
        
        # Setup Output Paths
        self.out_dir = self.config.target_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.outfile = self.out_dir / "trajectory.xyz"
        
        self.logger.info("Initialized Simulator.")
        self.logger.info(f"Output directory mapped to: {self.out_dir}")

    def _detect_hardware(self):
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        self.logger.info(f"Hardware detected! Routing tensors to: {device.upper()}")
        return device

    def _load_models(self):
        self.logger.info("Loading PyTorch MLIP Structure and Exchange models...")
        structuremodel = StructureGNN()
        exchangemodel = ExchangeMLP()

        structuremodel, _, _, _ = load_checkpoint(structuremodel, self.config.structurepath, self.device)
        exchangemodel, _, _, _ = load_checkpoint(exchangemodel, self.config.exchangepath, self.device)
        return structuremodel, exchangemodel

    def build_lattice(self):
        self.logger.info(f"Generating pristine CrI3 {self.config.nx}x{self.config.ny} lattice...")
        cri3_manager = CrI3()
        # Expand the unit cell into a supercell
        atoms = cri3_manager.batoms.copy() * (self.config.nx, self.config.ny, 1)
        return atoms
    
    def apply_strain(self, atoms):
        self.logger.info(f"Applying {self.config.strain_type} strain (Amplitude: {self.config.amplitude} Å)...")
        # Simplified StrainEngineer with quarter-wave logic removed
        strain_engineer = StrainEngineer(
            amplitude=self.config.amplitude, 
            strain_type=self.config.strain_type
        )
        
        # Route to the appropriate deformation method
        if self.config.strain_type in ["uniform_x", "uniform_y", "biaxial_uniform"]:
            # Example mapping: customize these string hooks based on your StrainEngineer updates
            strain_x = self.config.amplitude if "x" in self.config.strain_type or "biaxial" in self.config.strain_type else 0.0
            strain_y = self.config.amplitude if "y" in self.config.strain_type or "biaxial" in self.config.strain_type else 0.0
            strained_atoms = strain_engineer.apply_cell_strain(atoms, strain_x, strain_y, logger=self.logger)
        else:
            strained_atoms = strain_engineer.apply_ripple(atoms, logger=self.logger)
            
        return strained_atoms

    def run(self):
        # 1. Setup Structure & Calculator
        atoms = self.build_lattice()
        structuremodel, exchangemodel = self._load_models()
        atoms.calc = DSpinGNNCalculator(
            structuremodel=structuremodel, 
            exchangemodel=exchangemodel, 
            rcut=7.0, 
            device=self.device
        )

        # 2. Relax the Pristine Lattice
        self.logger.info("Optimizing lattice to the MLIP's exact equilibrium...")
        opt = FIRE(atoms, logfile=None) 
        opt.run(fmax=0.01) 
        self.logger.info("Optimization complete!")
        self.logger.info(f"Post-optimization lattice constants: {atoms.cell[0][0]:.3f} Å (X), {atoms.cell[1][1]:.3f} Å (Y)")
        
        # 3. Apply the Target Deformation
        atoms = self.apply_strain(atoms)
        atoms.set_array("Local_J", np.zeros(len(atoms)))

        # 4. Thermalize the System
        self.logger.info(f"Thermalizing to {self.config.tmpK}K...")
        MaxwellBoltzmannDistribution(atoms, temperature_K=self.config.tmpK)
        Stationary(atoms)
        ZeroRotation(atoms)

        # 5. Setup Langevin Dynamics
        dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=self.config.tmpK, friction=0.01)
        
        # Clear/Create the output file
        with open(self.outfile, 'w') as f:
            pass

        # 6. Attach MD Hooks (IO, Tracking, Progress Bar)
        def write_frame():
            if 'local_j' in atoms.calc.results:
                atoms.set_array("Local_J", atoms.calc.results['local_j'])
            write(self.outfile, atoms, format='extxyz', append=True)

        dyn.attach(write_frame, interval=5)
        
        self.logger.info("Initializing Maximum Force Tracker...")
        tracker = MaxForceTracker(atoms=atoms, dyn=dyn, xyz_path=str(self.outfile), logger=self.logger)
        dyn.attach(tracker, interval=10)

        pbar_interval = 10
        pbar = tqdm(total=self.config.timesteps, desc=f"MD Simulation ({self.config.tmpK}K)", unit="step", dynamic_ncols=True)
        dyn.attach(lambda: pbar.update(pbar_interval), interval=pbar_interval)

        # 7. Execute Simulation
        self.logger.info("Starting Simulation...")
        dyn.run(self.config.timesteps)
        
        pbar.close()
        self.logger.info(f"MD Finished! Open '{self.outfile}' in OVITO.")
        self.logger.info(f"Force logs saved to '{tracker.log_file}'.")


from logger import getlogger
logger = getlogger()
# =========================================================
# Execution Center
# =========================================================
if __name__ == "__main__":
    # 1. Define physics parameters cleanly using the DataClass
    config = SimConfig(
        structurepath="checkpoints/Structural/DOCheckpoints-Full-Exchange-DataSet1-1/Structure-Epoch-8800.pt",
        exchangepath="checkpoints/Exchange/Stripped-Data-ML-Analytical-16_Embeds_32-Weightdecay/Exchange-Epoch-33000.pt",
        nx=5,                     
        ny=5,                     
        tmpK=5,                  
        timesteps=10000,          
        amplitude=0.5,          
        strain_type="uniaxial" # Maps to density wave ripple or uniform strain depending on apply_strain hook
    )
    
    # 2. Initialize and run
    simulator = CrI3_Simulator(config)
    simulator.run()