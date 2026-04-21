import itertools
import torch

# ==========================================
# 2. Geometric Physics Module
# ==========================================
class GeometryExtractor:
    """Calculates all physical descriptors (supercells, angles, bond lengths)."""
    def __init__(self, device='cpu', ligand_symbol='I'):
        self.device = device
        self.ligand_symbol = ligand_symbol

    def build_ligand_supercell(self, positions, symbols, cell):
        """Creates a 3x3x1 virtual supercell of ligand atoms to find bridging paths."""
        ligand_indices = [idx for idx, sym in enumerate(symbols) if sym == self.ligand_symbol]
        ligand_pos = positions[ligand_indices]
        
        # 3x3x1 grid shifts
        shifts = torch.tensor(list(itertools.product([-1, 0, 1], [-1, 0, 1], [0])), 
                              dtype=torch.float32, device=self.device)
        shift_vecs = shifts @ cell
        all_ligand_pos = (ligand_pos.unsqueeze(0) + shift_vecs.unsqueeze(1)).view(-1, 3)
        return all_ligand_pos

    def get_angles_and_bonds(self, all_ligand_pos, ipos, fpos, cri_rcut=3.2):
        """Calculates the Cr-I-Cr angle and specific bond lengths."""
        dist_to_i = torch.norm(all_ligand_pos - ipos, dim=1)
        dist_to_j = torch.norm(all_ligand_pos - fpos, dim=1)

        # Bridging mask
        shared_mask = (dist_to_i < cri_rcut) & (dist_to_j < cri_rcut)
        shared_ligands = all_ligand_pos[shared_mask]

        if len(shared_ligands) > 0:
            cos_thetas = []
            local_bonds = []
            
            for k in range(len(shared_ligands)):
                Ik_pos = shared_ligands[k]
                v1 = ipos - Ik_pos
                v2 = fpos - Ik_pos
                
                cos_theta = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
                cos_thetas.append(cos_theta.item())
                
                local_bonds.extend([torch.norm(v1).item(), torch.norm(v2).item()])
            
            avg_cos_theta = sum(cos_thetas) / len(cos_thetas)
            cri_bonds = sorted(local_bonds)
        else:
            avg_cos_theta = 0.0 
            cri_bonds = []

        # BUG FIX: Ensure cri_bonds is ALWAYS exactly length 4 for the neural network
        while len(cri_bonds) < 4:
            cri_bonds.append(2.7) # Pad missing bonds with equilibrium
        if len(cri_bonds) > 4:
            cri_bonds = cri_bonds[:4] # Truncate excess (e.g. 3 bridging iodines)

        return avg_cos_theta, cri_bonds