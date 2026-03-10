from ase.io import read, Trajectory
from ase.md.langevin import Langevin
from ase import units
import torch

from .nequipcalc import NequIPCalculator
from trainutils import loadmodel

model = loadmodel('checkpoints/latest-model.pt', mps=True)

# 1. Load your verified system
atoms = read('aspirin.xyz')
atoms.calc = NequIPCalculator(model=model, device='mps')
# 2. Set initial velocities to 300K
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# 3. Setup the Thermostat (NVT)
# timestep: 0.5 fs is safe for C-H bonds
# friction: 0.01 is a standard choice for gas-phase-like dynamics
dyn = Langevin(atoms, timestep=0.5 * units.fs, temperature_K=300, friction=0.01)

# 4. Save the "Production" trajectory
traj = Trajectory('aspirin_production.traj', 'w', atoms)
dyn.attach(traj.write, interval=10) # Save every 10 steps to save disk space

# 5. Run for a longer duration (e.g., 10 picoseconds = 20,000 steps)
print("Starting Production MD...")
dyn.run(20000)
print("MD Finished!")