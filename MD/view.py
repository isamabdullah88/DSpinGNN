from ase.visualize import view
from ase.io import read

# View a single structure
atoms = read('aspirin.xyz')
view(atoms)

# OR view the entire trajectory as a movie
# traj = read('aspirin_production.traj', index=':')
# view(traj)