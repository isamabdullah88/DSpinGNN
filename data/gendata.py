import os
import torch
import numpy as np
from torch_geometric.data import Data

from .hubbard import EspressoHubbard
from .crI3 import CrI3
from .exchange import ExchangeGraph
from .crystaltensor import CrystalGraphTensor


class DataGenerator:
    def __init__(self, rcut, stntypes, strains, phase='FM'):
        self.rcut = rcut

        self.dataset = []

        self.phase = phase
        self.stntypes = stntypes
        self.strains = strains

        self.hubbard = EspressoHubbard()
        self.crI3 = CrI3()

        self.egraph = ExchangeGraph()
        self.cgraph = CrystalGraphTensor()

    


    def parse(self, wkdir, stntype, stnvalue, rattleidx):
        pwopath = os.path.join(wkdir, f"Strain_{stntype}_{stnvalue:.4f}_{rattleidx}", "espresso.pwo")
        if not os.path.exists(pwopath):
            print(f"Output file not found for strain {stnvalue:.4f} at {pwopath}. Skipping...")
            return None

        atoms = self.crI3.strain_atoms(stntype=stntype, stnvalue=stnvalue)
        atomsout = self.hubbard.parse(pwopath, atoms)

        z = torch.tensor(atomsout.get_atomic_numbers(), dtype=torch.long)
        pos = torch.tensor(atomsout.get_positions(), dtype=torch.float32)
        energy = atomsout.get_potential_energy()
        forces = atomsout.get_forces()
        cell = torch.tensor(atomsout.get_cell(), dtype=torch.float32)

        #  Parse TB2J outputs for this strain
        # tb2jpath = os.path.join(wkdir, f"Strain_{stntype}_{stnvalue:.4f}", "tmp/TB2J_results/Multibinit", "exchange.xml")
        # if not os.path.exists(tb2jpath):
        #     print(f"TB2J output file not found for strain {stnvalue:.4f} at {tb2jpath}. Skipping TB2J parsing...")
            
        edgeidxs, edgeshifts = self.cgraph.tensorgraph(self.rcut)
        # print('pos: ', pos.shape)
        # print('edges: ', edgeidxs.shape)
        # cr_edges, exchangejs, cr_shifts = self.egraph.graph(tb2jpath)

        data = Data(
            z=z,
            pos=pos,
            cell=cell,
            y_energy=torch.tensor([energy], dtype=torch.float32),
            y_forces=torch.tensor(forces, dtype=torch.float32),
            edge_index=edgeidxs,
            edge_shift=edgeshifts,
            # cr_edge_index=cr_edges,
            # cr_edge_shift=cr_shifts,
            # exchangejs=exchangejs
        )

        return data
    
    def generate(self, datasetdir, rattleidxs):
        """
        Main method to generate the dataset. It iterates over all strains, parses the DFT and TB2J outputs, and saves the PyG Data objects.
        """
        numsamples = 0
        for stntype, stnvalues in zip(self.stntypes, self.strains):
            wkdir = os.path.join(datasetdir, f"Rattle-{stntype}/{self.phase}")
            for strain in stnvalues:
                for rattleidx in rattleidxs:
                    print(f"Processing strain {strain:.4f} ({rattleidx+1}/{len(rattleidxs)})...")
                    
                    data = self.parse(wkdir, stntype, strain, rattleidx)
                    if data is not None:
                        self.dataset.append(data)
                        numsamples += 1
                    else:
                        print(f"Data parsing failed for strain {strain:.4f} (Rattle idx: {rattleidx}). Skipping this sample.")
            
            wkdir = os.path.join(datasetdir, f"Rattle-{stntype}-4/{self.phase}")
            for strain in stnvalues:
                for rattleidx in rattleidxs:
                    print(f"Processing strain {strain:.4f} ({rattleidx+1}/{len(rattleidxs)})...")
                    
                    data = self.parse(wkdir, stntype, strain, rattleidx)
                    if data is not None:
                        self.dataset.append(data)
                        numsamples += 1
                    else:
                        print(f"Data parsing failed for strain {strain:.4f} (Rattle idx: {rattleidx}). Skipping this sample.")

        print(f"Dataset generation complete. Total samples: {numsamples}")

        return self.dataset



if __name__ == "__main__":
    
    datasetdir = "./DataSets/GNN/"
    datasetpath = "./DataSets/GNN/RattleGNN-rcut_15.pth"

    RCUT = 15.0
    stntypes = ['Biaxial', 'Uniaxial_X', 'Shear_XY']
    strains = [[float(s) for s in np.linspace(-0.12, 0.12, 15) if -0.05 < s < 0.05]]
    strains += [[float(s) for s in np.linspace(-0.15, 0.15, 21) if -0.05 < s < 0.05]]
    print('strains: ', strains)
    rattleidxs = list(range(10))
    datagen = DataGenerator(RCUT, stntypes=stntypes, strains=strains, phase='FM')

    dataset = datagen.generate(datasetdir=datasetdir, rattleidxs=rattleidxs)

    print(f"Generated dataset with {len(dataset)} samples. Saving to {datasetpath}...")
    torch.save(dataset, datasetpath)

