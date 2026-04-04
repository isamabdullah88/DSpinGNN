import torch
import numpy as np
import matplotlib.pyplot as plt

from data.crI3 import CrI3
from .dspingnn import DSpinGNNCalculator
from model import DSpinGNN
from train.trainutils import load_checkpoint

# 1. Auto-Detect Hardware
if torch.cuda.is_available():
    device = 'cuda'
    use_mps = False
elif torch.backends.mps.is_available():
    device = 'mps'
    use_mps = True
else:
    device = 'cpu'
    use_mps = False

print(f"Hardware detected! Routing tensors to: {device.upper()}")

# 2. Load the Model
model = DSpinGNN(mps=use_mps)
model, _, _, _ = load_checkpoint(model, 'checkpoints/latest-model.pt', device)

# 3. Setup the CrI3 Manager
cri3_manager = CrI3() 

# 4. Define the Strain Grid
# -12% to +12% strain in 51 steps
strains = np.linspace(-0.12, 0.12, 51) 
strain_types = ['Biaxial', 'Uniaxial_X', 'Shear_XY']

# Setup the plot
plt.figure(figsize=(10, 7))
colors = {'Biaxial': 'blue', 'Uniaxial_X': 'green', 'Shear_XY': 'red'}

print("Calculating Equations of State...")

# 5. The Strain Loop
for stntype in strain_types:
    print(f"--> Processing {stntype}...")
    energies = []
    
    for s in strains:
        # Use YOUR class to generate the mathematically perfect strained cell!
        strained_atoms = cri3_manager.strain_atoms(stntype=stntype, stnvalue=s)
        
        # Attach the calculator to this specific strained state
        strained_atoms.calc = DSpinGNNCalculator(model=model, rcut=7.0, device=device)
        
        # Calculate and store the Total Energy
        energies.append(strained_atoms.get_potential_energy())
        
    # Offset energies so the minimum sits at 0 eV for easier visual comparison
    energies = np.array(energies)
    energies -= np.min(energies) 
        
    # Plot the curve
    plt.plot(strains * 100, energies, marker='o', 
             linewidth=2, color=colors[stntype], label=stntype)

# 6. Polish and Display the Graph
plt.xlabel("Strain (%)", fontsize=14)
plt.ylabel("Relative Total Energy (eV)", fontsize=14)
plt.title("DSpinGNN: Elastic Strain Paraboloids", fontsize=16)
plt.axvline(0, color='black', linestyle='--', alpha=0.5) # Mark 0% Strain
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

print("Calculations complete! Opening plot...")
plt.show()