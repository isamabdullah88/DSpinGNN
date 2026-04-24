import itertools
import torch
import logging

class GeometryExtractor:
    """Calculates all physical descriptors (supercells, angles, bond lengths)."""
    def __init__(self, device='cpu', lsymbol='I'):
        # lsymbol is the ligand symbol (e.g., 'I' for iodide)
        self.device = device
        self.lsymbol = lsymbol

        self.logger = logging.getLogger(__name__)

    def ligandscell(self, positions, symbols, cell):
        """Creates a 3x3x1 virtual supercell of ligand atoms to find bridging paths."""
        idxs = [idx for idx, sym in enumerate(symbols) if sym == self.lsymbol]
        lpositions = positions[idxs]
        
        # 3x3x1 grid shifts
        shifts = torch.tensor(list(itertools.product([-1, 0, 1], [-1, 0, 1], [0])), 
                              dtype=torch.float32, device=self.device)
        shiftvecs = shifts @ cell
        alllpositions = (lpositions.unsqueeze(0) + shiftvecs.unsqueeze(1)).view(-1, 3)
        return alllpositions

    def calc_bondsangles(self, lpositions, ipos, fpos, lrcut=3.2):
        """
        Calculates the Cr-I-Cr angle and specific Cr-I bond lengths.
        lrcut: Ligand cutoff distance to consider a ligand as bridging.
        """
        disti = torch.norm(lpositions - ipos, dim=1)
        distj = torch.norm(lpositions - fpos, dim=1)

        # Bridging mask
        mask = (disti < lrcut) & (distj < lrcut)
        sligands = lpositions[mask]

        if len(sligands) > 0:
            cosangles = []
            lbonds = []
            angles = []
            
            for k in range(len(sligands)):
                kpos = sligands[k]
                v1 = ipos - kpos
                v2 = fpos - kpos
                
                cosangle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8)

                if torch.norm(v1) < 0.1 or torch.norm(v2) < 0.1:
                    self.logger.info(f"Norm v1: {torch.norm(v1).item():.3f}, Norm v2: {torch.norm(v2).item():.3f}")
        
                # Apply inverse cosine immediately and convert to degrees
                cosangle_clamped = torch.clamp(cosangle, min=-1.0, max=1.0)
                angle = torch.rad2deg(torch.acos(cosangle_clamped))
                
                if torch.isnan(angle):
                    self.logger.info(f"Warning: NaN angle detected. Cosine value: {cosangle.item():.4f}")
                    self.logger.info(f"Angles: {angle.item():.2f} degrees, Cosine: {cosangle.item():.4f}")

                cosangles.append(cosangle.item())
                angles.append(angle.item())
                
                lbonds.extend([torch.norm(v1).item(), torch.norm(v2).item()])
            
            avgangle = (sum(angles) / len(angles)) if len(angles) > 0 else 0.0
            avgcosangle = (sum(cosangles) / len(cosangles)) if len(cosangles) > 0 else 0.0
            bonds = sorted(lbonds)
        else:
            avgcosangle = 0.0 
            bonds = []
            avgangle = 0.0

        # Ensure 4 bond lengths
        while len(bonds) < 4:
            bonds.append(2.7)

        # Strip to 4 bond lengths if more are present
        if len(bonds) > 4:
            bonds = bonds[:4]

        return avgcosangle, bonds, avgangle