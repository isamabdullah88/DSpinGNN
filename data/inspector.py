import torch
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 2. Analytics & Visualization Module
# ==========================================
class GraphVisualizer:
    """Extracts physical tensors from dataloaders and plots distributions."""
    
    @staticmethod
    def _calculate_pbc_distances(batch):
        """DRY Helper method to calculate accurate distances across PBC boundaries."""
        src, dst = batch.cr_edge_index
        graphidxs = batch.batch[src]
        
        cell = batch.cell.view(-1, 3, 3)
        bcell = cell[graphidxs]
        
        edge_shift = batch.cr_edge_shift.to(bcell.dtype)
        tvec = torch.einsum('ei, eij -> ej', edge_shift, bcell)
        
        radvec = batch.pos[dst] - batch.pos[src] + tvec
        return radvec.norm(dim=1, keepdim=False)

    def plot_distances(self, dataloader):
        all_cr_cr, all_cr_i = [], []
        
        with torch.no_grad():
            for batch in dataloader:
                # 1. Extract Cr-I Legs
                # Note: Ensure 'cr_i_bonds' matches your new pipeline (e.g., 'cri_bonds_tensor')
                all_cr_i.extend(batch.cr_i_bonds.view(-1).cpu().numpy())
                
                # 2. Extract Cr-Cr distances
                dist = self._calculate_pbc_distances(batch)
                all_cr_cr.extend(dist.cpu().numpy())
        
        all_cr_cr = np.array(all_cr_cr)
        all_cr_i = np.array(all_cr_i)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.hist(all_cr_cr, bins=50, color='purple', edgecolor='black', alpha=0.7)
        ax1.set_title("Distribution of Cr-Cr Distances")
        ax1.set_xlabel("Distance (Å)")
        ax1.set_ylabel("Frequency")
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(all_cr_i, bins=70, color='teal', edgecolor='black', alpha=0.7)
        ax2.set_title("Distribution of Individual Cr-I Legs\n(4 per Cr-Cr edge)")
        ax2.set_xlabel("Distance (Å)")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        print("-" * 30)
        print(f"Cr-Cr Range: {all_cr_cr.min():.4f} to {all_cr_cr.max():.4f} Å")
        print(f"Cr-I  Range: {all_cr_i.min():.4f} to {all_cr_i.max():.4f} Å")
        print("-" * 30)

    def plot_cosines(self, dataloader):
        all_cosines = []
        with torch.no_grad():
            for batch in dataloader:
                all_cosines.extend(batch.cr_cr_angles.view(-1).cpu().numpy())
        
        plt.figure(figsize=(8, 5))
        plt.plot(all_cosines, marker='o', linestyle='', markersize=2)
        plt.title("Distribution of Cosine Angles in Dataset")
        plt.xlabel("Cosine of Cr-Cr-Cr Angle")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

    def plot_j_vs_distance(self, dataloader, save_path="j_vs_distance_distribution.png"):
        all_distances, all_j_values = [], []
        
        with torch.no_grad():
            for batch in dataloader:
                dist = self._calculate_pbc_distances(batch)
                yexchange = batch.y_exchange.view(-1)
                
                all_distances.extend(dist.cpu().numpy())
                all_j_values.extend(yexchange.cpu().numpy())

        plt.figure(figsize=(10, 6))
        plt.scatter(all_distances, all_j_values, alpha=0.3, s=15, c='blue', edgecolors='none')
        plt.axvline(x=4.0, color='red', linestyle='--', linewidth=2, label='4.0 Å Cutoff (1NN)')
        plt.axhline(y=0.0, color='black', linestyle='-', linewidth=1)
        
        plt.title("Cr-Cr Exchange Coupling vs. Bond Distance", fontsize=16)
        plt.xlabel("Cr-Cr Bond Distance (Å)", fontsize=14)
        plt.ylabel("Exchange Coupling J Magnitude", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')