import torch
import xml.etree.ElementTree as ET
import itertools
from .CrI3 import CrI3 # Assuming this is your crystal builder class

class ExchangeGraph:
    def __init__(self):
        self.crystal = CrI3()

    def parseTB2J(self, xmlpath):
        """Parses a TB2J XML output file to extract exchange interactions."""
        tree = ET.parse(xmlpath)
        root = tree.getroot()
        
        TB2Jdict = {}
        spinexchange_list = root.find('spin_exchange_list')
        
        for interaction in spinexchange_list.findall('spin_exchange_term'):
            ijR = interaction.find('ijR').text.split()
            i, j, Rx, Ry, Rz = int(ijR[0]), int(ijR[1]), int(ijR[2]), int(ijR[3]), int(ijR[4])
            
            # Convert to meV
            jval = float(interaction.find('data').text.split(' ')[0]) * 1000.0
            TB2Jdict[i, j, Rx, Ry, Rz] = jval
        
        return TB2Jdict

    def graph(self, xmlpath, rcut, atoms):
        """Converts an ASE Atoms object and TB2J outputs into PyG tensors."""
        scell = torch.tensor(atoms.get_cell(), dtype=torch.float32)
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        symbols = atoms.get_chemical_symbols()

        # ==========================================
        # PRE-COMPUTE: The 2D Iodine Virtual Supercell
        # ==========================================
        iodine_indices = [idx for idx, sym in enumerate(symbols) if sym == 'I']
        iodine_pos = positions[iodine_indices]
        
        # Optimized for Monolayer (No Z-axis expansion to save compute)
        shifts = torch.tensor(list(itertools.product([-1, 0, 1], [-1, 0, 1], [0])), dtype=torch.float32)
        shift_vecs = shifts @ scell
        all_iodine_pos = (iodine_pos.unsqueeze(0) + shift_vecs.unsqueeze(1)).view(-1, 3)

        TB2Jdict = self.parseTB2J(xmlpath)
        
        cr_edges, exchangejs, edgeshifts, edgedists, cr_cr_angles = [], [], [], [], []
        cr_i_mins = [] # NEW GLOBAL LIST
        cr_i_maxs = [] # NEW GLOBAL LIST
        
        for item in TB2Jdict.items():
            (i, j, Rx, Ry, Rz), jval = item
            i -= 1 
            j -= 1

            ipos = positions[i]
            fpos = positions[j] + (Rx * scell[0] + Ry * scell[1] + Rz * scell[2])
            dist = ipos.dist(fpos)
            
            if dist > rcut:
                continue

            # ==========================================
            # PHYSICS INJECTION: Calculate Cr-I-Cr Angle
            # ==========================================
            dist_to_i = torch.norm(all_iodine_pos - ipos, dim=1)
            dist_to_j = torch.norm(all_iodine_pos - fpos, dim=1)

            # Bridging mask (< 3.2 Å is safe for strained CrI3)
            shared_mask = (dist_to_i < 3.2) & (dist_to_j < 3.2)
            shared_iodines = all_iodine_pos[shared_mask]

            if len(shared_iodines) > 0:
                cos_thetas = []
                local_cri_mins = [] # Renamed to avoid confusion!
                local_cri_maxs = [] 
                
                for k in range(len(shared_iodines)):
                    Ik_pos = shared_iodines[k]
                    v1 = ipos - Ik_pos
                    v2 = fpos - Ik_pos
                    
                    cos_theta = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
                    cos_thetas.append(cos_theta.item())
                    
                    d1 = torch.norm(v1).item()
                    d2 = torch.norm(v2).item()
                    local_cri_mins.append(min(d1, d2))
                    local_cri_maxs.append(max(d1, d2))
                
                avg_cos_theta = sum(cos_thetas) / len(cos_thetas)
                avg_cri_min = sum(local_cri_mins) / len(local_cri_mins)
                avg_cri_max = sum(local_cri_maxs) / len(local_cri_maxs)
            else:
                avg_cos_theta = 0.0 
                avg_cri_min = 2.7 
                avg_cri_max = 2.7

            cr_edges.append([i, j])
            exchangejs.append(jval)
            edgeshifts.append([Rx, Ry, Rz])
            edgedists.append(dist.item())
            cr_cr_angles.append(avg_cos_theta)
            
            cr_i_mins.append(avg_cri_min) # <-- Append the scalar!
            cr_i_maxs.append(avg_cri_max) # <-- Append the scalar!

        # Convert to final PyTorch tensors
        cr_edges = torch.tensor(cr_edges, dtype=torch.long).t().contiguous()
        exchangejs = torch.tensor(exchangejs, dtype=torch.float32).view(-1, 1)
        edgeshifts = torch.tensor(edgeshifts, dtype=torch.int64)
        edgedists = torch.tensor(edgedists, dtype=torch.float32)
        cr_cr_angles = torch.tensor(cr_cr_angles, dtype=torch.float32).view(-1, 1)

        # Ensure all physical features are shape [num_edges, 1]
        cr_cr_angles = torch.tensor(cr_cr_angles, dtype=torch.float32).view(-1, 1)
        cr_i_mins = torch.tensor(cr_i_mins, dtype=torch.float32).view(-1, 1)
        cr_i_maxs = torch.tensor(cr_i_maxs, dtype=torch.float32).view(-1, 1)

        return cr_edges, exchangejs, edgeshifts, edgedists, cr_cr_angles, cr_i_mins, cr_i_maxs