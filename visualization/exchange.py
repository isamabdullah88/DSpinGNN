import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

def plot_j_field(xyz_file, frame_index=-1):
    print(f"Loading frame {frame_index} from {xyz_file}...")
    
    # Read the specific frame from the trajectory (-1 gets the final frame)
    try:
        atoms = read(xyz_file, index=frame_index)
    except Exception as e:
        print(f"Failed to load file: {e}")
        return

    # Extract standard ASE properties
    positions = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()

    # Extract our custom neural network array
    try:
        local_j = atoms.get_array("Local_J")
    except KeyError:
        print("\n[ERROR] 'Local_J' array not found in the XYZ file!")
        print("Double-check that your NVTensemble script has:")
        print("1. atoms.set_array('Local_J', np.zeros(len(atoms)))")
        print("2. write(..., format='extxyz')")
        return

    # Filter for Chromium atoms (Z = 24)
    # If we leave Iodine (Local_J = 0.0), it will ruin the color gradient scale
    cr_mask = (atomic_numbers == 24)
    cr_positions = positions[cr_mask]
    cr_j_values = local_j[cr_mask]
    
    print(f"Found {len(cr_j_values)} Chromium atoms.")
    print(f"J Value Range: [{cr_j_values.min():.4f} to {cr_j_values.max():.4f}]")

    # ==========================================
    # PLOTTING
    # ==========================================
    plt.figure(figsize=(10, 8))
    
    # Extract X and Y coordinates for the 2D top-down view
    x = cr_positions[:, 0]
    y = cr_positions[:, 1]

    # Create a scatter plot colored by Local_J
    # 'coolwarm' maps the lowest values to blue and highest to red
    scatter = plt.scatter(
        x, y, 
        c=cr_j_values, 
        cmap='coolwarm', 
        s=300,             # Dot size
        edgecolors='black', # Clean borders around the atoms
        zorder=2
    )
    
    # Attach the colorbar to decode the values
    cbar = plt.colorbar(scatter)
    cbar.set_label('Local Exchange J', fontsize=14, fontweight='bold')

    # Formatting
    plt.title('CrI3 Magnetic Exchange Field (Top-Down View)', fontsize=16, fontweight='bold')
    plt.xlabel('X Position (Å)', fontsize=14)
    plt.ylabel('Y Position (Å)', fontsize=14)
    
    # Force axes to be equal so the hexagonal lattice doesn't look stretched
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5, zorder=1)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Update this to match your actual output filename!
    filename = "CrI3-MD-MixedData-3-2.xyz" 
    
    # Plot the very last frame of the simulation
    plot_j_field(filename, frame_index=-1)