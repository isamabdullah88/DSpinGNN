import matplotlib.pyplot as plt
import numpy as np

def plot_md_energies(steps, epot, ekin, filename="md_stability.png"):
    etot = np.array(epot) + np.array(ekin)
    time = np.array(steps) * 0.5  # Assuming 0.5 fs timestep
    
    plt.figure(figsize=(10, 6))
    
    # Plot individual components
    plt.plot(time, epot, label='Potential Energy', color='blue', alpha=0.6)
    plt.plot(time, ekin, label='Kinetic Energy', color='orange', alpha=0.6)
    
    # Plot Total Energy (the verification line)
    plt.plot(time, etot, label='Total Energy', color='black', linewidth=2)
    
    plt.xlabel('Time (fs)')
    plt.ylabel('Energy (eV)')
    plt.title('NVE Energy Conservation - Aspirin')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Zoom in on Total Energy in an inset to show the tiny fluctuations
    # This is a common requirement for thesis papers
    ax = plt.gca()
    inset = ax.inset_axes([0.5, 0.1, 0.4, 0.3])
    inset.plot(time, etot, color='black')
    inset.set_title('Total Energy Zoom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Example usage:
# plot_md_energies(range(len(epot_list)), epot_list, ekin_list)

from ase.io import read
from ase.visualize import view

# Load the trajectory file you saved
traj = read('aspirin_md.traj', index=':')

# This opens a pop-up window where you can play the animation
view(traj)

from ase.io import read
from ase.visualize.plot import plot_atoms
import matplotlib.pyplot as plt

traj = read('aspirin_md.traj', index=':')
# Pick 4 snapshots: start, 1/3, 2/3, and end
snapshots = [traj[0], traj[len(traj)//3], traj[2*len(traj)//3], traj[-1]]

fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for i, atoms in enumerate(snapshots):
    plot_atoms(atoms, ax[i], rotation=('0x,0y,0z'))
    ax[i].set_title(f"Frame {i}")
    ax[i].axis('off')

plt.show()

from ase import units

def plot_temperature(ekin, num_atoms):
    # KE is in eV, we need T in Kelvin
    # formula: T = 2 * ekin / (3 * num_atoms * units.kB)
    temperatures = [2 * k / (3 * num_atoms * units.kB) for k in ekin]
    
    plt.figure(figsize=(8, 4))
    plt.plot(temperatures, color='red')
    plt.axhline(y=np.mean(temperatures), color='black', linestyle='--')
    plt.ylabel('Temperature (K)')
    plt.xlabel('Steps')
    plt.title(f'Mean Temperature: {np.mean(temperatures):.2f} K')
    plt.show()