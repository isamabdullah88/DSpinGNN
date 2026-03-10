import numpy as np
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.md import MDLogger

from trainutils import loadmodel
from MD.nequipcalc import NequIPCalculator

# 1. Setup atoms and velocities
atoms = read('aspirin.xyz')
from ase.visualize.plot import plot_atoms
import matplotlib.pyplot as plt

# Initialize velocities for 300K
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# 2. Attach your custom calculator
# (Assuming your MyNequIPCalculator class is defined as before)
model = loadmodel('./checkpoints/latest-model.pt', mps=True)
calc = NequIPCalculator(model=model, device='mps')
atoms.calc = calc

# 3. Setup NVE Dynamics (Velocity Verlet)
# Note: Use a small timestep for organic molecules (0.5 fs)
dt = 0.5 * units.fs
dyn = VelocityVerlet(atoms, dt)

# 4. Energy Monitoring Logic
energies = []

def log_energy():
    epot = atoms.get_potential_energy()
    ekin = atoms.get_kinetic_energy()
    etot = epot + ekin
    energies.append(etot)
    print(f"Step: {len(energies)} | TotEnergy: {etot:.6f} eV | Temp: {atoms.get_temperature():.2f} K")

dyn.attach(log_energy, interval=1)

# Run for 1000 steps (~0.5 picoseconds)
dyn.run(1000)

# 5. Quantify Drift
energy_array = np.array(energies)
drift = (energy_array[-1] - energy_array[0]) / (len(energy_array) * 0.5) # eV / fs
print(f"\nFinal Energy Drift: {drift:.2e} eV/fs")