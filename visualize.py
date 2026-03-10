import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform

def visualize_rmd17_frame(npz_path, frame_idx=0, scale_forces=0.5):
    """
    Visualizes a single frame of the rMD17 dataset including bonds and force vectors.
    """
    
    # 1. Load Data
    try:
        data = np.load(npz_path)
        R = data['R'][frame_idx]  # Positions (N_atoms, 3)
        F = data['F'][frame_idx]  # Forces (N_atoms, 3)
        z = data['z']             # Atomic numbers (N_atoms,)
    except FileNotFoundError:
        print("Dataset not found. Generating dummy Aspirin data for demo...")
        # Dummy data roughly approximating an Aspirin-like structure for testing
        R = np.random.rand(21, 3) * 4 
        F = (np.random.rand(21, 3) - 0.5) * 5
        z = np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1])

    # 2. Setup Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 3. Define Atom Aesthetics (CPK Coloring)
    colors = {
        1: 'lightgray',  # Hydrogen
        6: 'black',      # Carbon
        8: 'red'         # Oxygen
    }
    atom_sizes = {1: 100, 6: 300, 8: 300}
    
    # 4. Draw Bonds (Simple Distance Heuristic)
    # Calculate all pairwise distances
    dist_matrix = squareform(pdist(R))
    
    # If atoms are closer than 1.6 Angstroms, draw a line (bond)
    # This works well for organic molecules like Aspirin
    bond_cutoff = 1.6 
    num_atoms = len(z)
    
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if dist_matrix[i, j] < bond_cutoff:
                ax.plot([R[i, 0], R[j, 0]], 
                        [R[i, 1], R[j, 1]], 
                        [R[i, 2], R[j, 2]], 
                        color='gray', linewidth=2, alpha=0.5)

    # 5. Draw Atoms
    # We plot them one by one to handle colors correctly
    for i in range(num_atoms):
        c = colors.get(z[i], 'blue')
        s = atom_sizes.get(z[i], 100)
        ax.scatter(R[i, 0], R[i, 1], R[i, 2], 
                   c=c, s=s, edgecolors='black', alpha=1.0, depthshade=False)

    # 6. Draw Forces (Quiver)
    # We project the force vector starting from the atom position
    # scale_forces helps visualize small/large forces appropriately
    ax.quiver(R[:, 0], R[:, 1], R[:, 2], 
              F[:, 0], F[:, 1], F[:, 2], 
              length=scale_forces, normalize=False, color='orange', linewidth=2, arrow_length_ratio=0.2)

    # 7. Final Polish
    ax.set_title(f"rMD17 Aspirin - Frame {frame_idx}\nOrange Arrows = Forces", fontsize=14)
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    
    # Center the view
    center = R.mean(axis=0)
    ax.set_xlim(center[0]-3, center[0]+3)
    ax.set_ylim(center[1]-3, center[1]+3)
    ax.set_zlim(center[2]-3, center[2]+3)
    
    plt.show()

# --- Run the function ---
# Replace 'rmd17_aspirin.npz' with your actual file path
visualize_rmd17_frame('Data/md17_aspirin.npz', frame_idx=100)