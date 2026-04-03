import torch
import xml.etree.ElementTree as ET

from .crI3 import CrI3

class ExchangeGraph:
    def __init__(self):
        self.crystal = CrI3()
        # self.atoms = self.crystal.batoms
        # self.scell = torch.tensor(self.atoms.scell, dtype=torch.float32)
        # self.pos = torch.tensor(self.atoms.get_positions(), dtype=torch.float32)

    def parseTB2J(self, xmlpath):
        """
        Parses a TB2J XML output file to extract exchange interactions.
        Returns a dictionary of the form: {(atom_i, atom_j): J_value_in_meV}
        """
        
        tree = ET.parse(xmlpath)
        root = tree.getroot()
        
        TB2Jdict = {}
        
        spinexchange_list = root.find('spin_exchange_list')
        # print('list: ', spinexchange_list)
        for interaction in spinexchange_list.findall('spin_exchange_term'):
            # print('interaction: ', interaction)
            ijR = interaction.find('ijR').text.split(' ')
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

        # satoms = self.crystal.strain_atoms(stntype=stntype, stnvalue=stnvalue)
        scell = torch.tensor(atoms.get_cell(), dtype=torch.float32)
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float32)

        TB2Jdict = self.parseTB2J(xmlpath)
        
        # z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        # pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        
        # Cr-Cr edges and exchange J values
        cr_edges = []
        exchangejs = []
        edgeshifts = []
        edgedists = []
        print('rcut: ', rcut)
        
        for item in TB2Jdict.items():
            (i, j, Rx, Ry, Rz), jval = item

            i -= 1 # Convert to 0-based indexing
            j -= 1

            # print('i, j: Rx, Ry, Rz', i, j, Rx, Ry, Rz)
            ipos = positions[i]
            fpos = positions[j] + (Rx * scell[0] + Ry * scell[1] + Rz * scell[2])
            if ipos.dist(fpos) > rcut:
                # print('i,j: ', i, j, 'distance: ', ipos.dist(fpos), 'exceeds cutoff: ', rcut)
                continue

            edgedists.append(ipos.dist(fpos).item())
            edgedists.append(ipos.dist(fpos).item()) # Symmetric edge

            cr_edges.append([i, j])
            cr_edges.append([j, i]) # Undirected/symmetric edge
            
            # jval = TB2Jdict[(i, j), R]
            exchangejs.append(jval)
            exchangejs.append(jval) # Symmetric J

            edgeshifts.append([Rx, Ry, Rz])
            edgeshifts.append([-Rx, -Ry, -Rz]) # Shift for symmetric edge

            
        # import matplotlib.pyplot as plt
        # plt.plot(dists, exchangejs[:len(dists)], 'o')
        # plt.xlabel('Distance (Å)')
        # plt.ylabel('Exchange J (meV)')
        # plt.title(f'Exchange J vs Distance for Strain {stnvalue:.4f}')
        # plt.grid()
        # plt.show()
        cr_edges = torch.tensor(cr_edges, dtype=torch.long).t().contiguous()
        exchangejs = torch.tensor(exchangejs, dtype=torch.float32).view(-1, 1) # Target J values
        edgeshifts = torch.tensor(edgeshifts, dtype=torch.int64) # Edge shifts for periodicity
        edgedists = torch.tensor(edgedists, dtype=torch.float32)
        
        # print('cr_edges: ', cr_edges.dtype, cr_edges.shape)
        # print('exchangejs: ', exchangejs.dtype, exchangejs.shape)
        # print('edgeshifts: ', edgeshifts.dtype, edgeshifts.shape)

        # 3. Build the PyG Data Object
        # data = Data(
        #     z = z, 
        #     pos = pos, 
            
        #     y_energy = torch.tensor([energy], dtype=torch.float32),
        #     y_forces = torch.tensor(forces, dtype=torch.float32),
            
        #     # Magnetic Exchange Targets (Edge Level) for edge decoder
        #     edge_index = cr_edges,
        #     edge_shift = torch.tensor(edgeshifts, dtype=torch.float32),
        #     exchangejs = exchangejs
        # )
        
        return cr_edges, exchangejs, edgeshifts, edgedists