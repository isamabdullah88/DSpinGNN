import torch
import xml.etree.ElementTree as ET

from .crI3 import CrI3

class ExchangeGraph:
    def __init__(self):
        self.crystal = CrI3()

    def parseTB2J(self, xmlpath):
        """
        Parses a TB2J XML output file to extract exchange interactions.
        Returns a dictionary of the form: {(atom_i, atom_j): J_value_in_meV}
        """
        
        tree = ET.parse(xmlpath)
        root = tree.getroot()
        
        TB2Jdict = {}
        
        spinexchange_list = root.find('spin_exchange_list')
        
        for interaction in spinexchange_list.findall('spin_exchange_term'):
            ijR = interaction.find('ijR').text.split()
            i = int(ijR[0])
            j = int(ijR[1])
            Rx = int(ijR[2])
            Ry = int(ijR[3])
            Rz = int(ijR[4])

            jval = float(interaction.find('data').text.split(' ')[0])*1000.0
            
            TB2Jdict[i, j, Rx, Ry, Rz] = jval
        
        return TB2Jdict

    def graph(self, xmlpath, rcut, atoms):
        """
        Converts an ASE Atoms object and TB2J outputs into a PyG Data object.
        TB2Jdict should be a parsed dictionary of your TB2J exchange.out
        e.g., TB2Jdict = {(atom_i, atom_j): J_value_in_meV}
        """

        scell = torch.tensor(atoms.get_cell(), dtype=torch.float32)
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float32)

        TB2Jdict = self.parseTB2J(xmlpath)
        
        # Cr-Cr edges and exchange J values
        cr_edges = []
        exchangejs = []
        edgeshifts = []
        edgedists = []
        
        for item in TB2Jdict.items():
            (i, j, Rx, Ry, Rz), jval = item

            i -= 1 # Convert to 0-based indexing
            j -= 1

            ipos = positions[i]
            fpos = positions[j] + (Rx * scell[0] + Ry * scell[1] + Rz * scell[2])
            if ipos.dist(fpos) > rcut:
                continue

            edgedists.append(ipos.dist(fpos).item())
            cr_edges.append([i, j])
            exchangejs.append(jval)
            edgeshifts.append([Rx, Ry, Rz])

        cr_edges = torch.tensor(cr_edges, dtype=torch.long).t().contiguous()
        exchangejs = torch.tensor(exchangejs, dtype=torch.float32).view(-1, 1) # Target J values
        edgeshifts = torch.tensor(edgeshifts, dtype=torch.int64) # Edge shifts for periodicity
        edgedists = torch.tensor(edgedists, dtype=torch.float32)

        return cr_edges, exchangejs, edgeshifts, edgedists