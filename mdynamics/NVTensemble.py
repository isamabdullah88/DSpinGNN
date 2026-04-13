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
    
    def __init__(self, modelpath, nx, ny, tmpK, timesteps, amplitude, strain_type, 
                 qx=None, qy=None, qc=None, out_dir="Simulations"):
        self.modelpath = modelpath
        self.nx = nx
        self.ny = ny
        self.tmpK = tmpK
        self.timesteps = timesteps
        self.amplitude = amplitude
        self.strain_type = strain_type
        
        # Quarter-wave parameters
        self.qx = qx
        self.qy = qy
        self.qc = qc
        
        self.logger = self._setup_logger()
        self.device = self._detect_hardware()
        
        # =========================================================
        # DYNAMIC PATH AND NESTED DIRECTORY GENERATION
        # =========================================================
        model_file = os.path.basename(self.modelpath)
        self.model_name = os.path.splitext(model_file)[0]
        model_folder_name = os.path.basename(os.path.dirname(self.modelpath))
        
        # Build an elegant string for the folder name based on provided quarter-waves
        q_tags = []
        if self.qx is not None: q_tags.append(f"Qx{self.qx}")
        if self.qy is not None: q_tags.append(f"Qy{self.qy}")
        if self.qc is not None: q_tags.append(f"Qc{self.qc}")
        q_str = "_".join(q_tags) if q_tags else "QAuto"
        
        sim_folder_name = f"{self.strain_type}_{self.nx}x{self.ny}_{q_str}_T{self.tmpK}K_Amp{self.amplitude}A_{self.timesteps}steps"
        
        self.target_out_dir = os.path.join(out_dir, model_folder_name, sim_folder_name)
        os.makedirs(self.target_out_dir, exist_ok=True)
        self.outfile = os.path.join(self.target_out_dir, "trajectory.xyz")
        
        self.logger.info(f"Initialized Simulator.")
        self.logger.info(f"Directory mapped to: {self.target_out_dir}")

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
        self.logger.info(f"Generating pristine CrI3 {self.nx}x{self.ny} lattice...")
        cri3_manager = CrI3()
        patoms = cri3_manager.batoms.copy()
        atoms = patoms * (self.nx, self.ny, 1)
        return atoms
    
    def apply_strain(self, atoms):
        # Pass the quarter-wave parameters to the engineer
        strain_engineer = StrainEngineer(
            amplitude=self.amplitude, 
            strain_type=self.strain_type,
            qx=self.qx,
            qy=self.qy,
            qc=self.qc
        )
        strained_atoms = strain_engineer.apply_ripple(atoms, logger=self.logger)
        return strained_atoms

    def run(self):
        atoms = self.build_lattice()
        model = self._load_model()
        atoms.calc = DSpinGNNCalculator(model=model, rcut=7.0, device=self.device)

        self.logger.info("Optimizing lattice to the MLIP's exact equilibrium...")
        opt = FIRE(atoms, logfile=None) 
        opt.run(fmax=0.01) 
        self.logger.info("Optimization complete!")
        self.logger.info(f"Post-optimization lattice constants: {atoms.cell[0][0]:.3f} Å (X), {atoms.cell[1][1]:.3f} Å (Y)")
        
        atoms = self.apply_strain(atoms)
        atoms.set_array("Local_J", np.zeros(len(atoms)))

        self.logger.info(f"Thermalizing to {self.tmpK}K...")
        MaxwellBoltzmannDistribution(atoms, temperature_K=self.tmpK)
        Stationary(atoms)
        ZeroRotation(atoms)

        dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=self.tmpK, friction=0.01)
        open(self.outfile, 'w').close()

        def write_frame():
            if 'local_j' in atoms.calc.results:
                atoms.set_array("Local_J", atoms.calc.results['local_j'])
            write(self.outfile, atoms, format='extxyz', append=True)

        dyn.attach(write_frame, interval=5)
        
        self.logger.info("Initializing Maximum Force Tracker...")
        tracker = MaxForceTracker(atoms=atoms, dyn=dyn, xyz_path=self.outfile, logger=self.logger)
        dyn.attach(tracker, interval=10)

        pbar_interval = 10
        pbar = tqdm(total=self.timesteps, desc=f"MD Simulation ({self.tmpK}K)", unit="step", dynamic_ncols=True)

        def update_pbar():
            pbar.update(pbar_interval)

        dyn.attach(update_pbar, interval=pbar_interval)

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
        "modelpath": "checkpoints/DOCheckpoints-Full-Exchange-DataSet1-1/Epoch-9900.pt",
        "nx": 20,                     
        "ny": 20,                     
        "tmpK": 5,                  
        "timesteps": 500,          
        "amplitude": 4.5,          
        "strain_type": "uniaxial",
        
        # Quarter-wave multipliers. 
        # MUST use multiples of 4 (4, 8, 12) to safely complete a full wave and avoid boundary shattering.
        "qx": 4,
        "qy": 4,
        "qc": 4  # 4 = Exactly 1 full radial wave from center to edge
    }
    
    # Initialize and run
    simulator = CrI3_Simulator(**config)
    simulator.run()