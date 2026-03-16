import os
import torch
import numpy as np
from torch_geometric.data import Data

from .hubbard import EspressoHubbard
from .CrI3 import CrI3
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

    


    def parse(self, wkdir, stntype, stnvalue):
        pwopath = os.path.join(wkdir, f"Strain_{stntype}_{stnvalue:.4f}", "espresso.pwo")
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
        tb2jpath = os.path.join(wkdir, f"Strain_{stntype}_{stnvalue:.4f}", "tmp/TB2J_results/Multibinit", "exchange.xml")
        if not os.path.exists(tb2jpath):
            print(f"TB2J output file not found for strain {stnvalue:.4f} at {tb2jpath}. Skipping TB2J parsing...")
            
        edgeidxs, edgeshifts = self.cgraph.tensorgraph(rcut = self.rcut)
        cr_edges, exchangejs, cr_shifts = self.egraph.graph(tb2jpath)

        data = Data(
            z=z,
            pos=pos,
            cell=cell,
            y_energy=torch.tensor([energy], dtype=torch.float32),
            y_forces=torch.tensor(forces, dtype=torch.float32),
            edge_index=edgeidxs,
            edge_shift=edgeshifts,
            cr_edge_index=cr_edges,
            cr_edge_shift=cr_shifts,
            exchangejs=exchangejs
        )

        return data
    
    def generate(self, datasetdir):
        """
        Main method to generate the dataset. It iterates over all strains, parses the DFT and TB2J outputs, and saves the PyG Data objects.
        """
        for stntype, stnvalues in zip(self.stntypes, self.strains):
            wkdir = os.path.join(datasetdir, f"Exchange-{stntype}/{self.phase}")
            for i, strain in enumerate(stnvalues):
                print(f"Processing strain {strain:.4f} ({i+1}/{len(stnvalues)})...")
                data = self.parse(wkdir, stntype, strain)
                if data is not None:
                    self.dataset.append(data)

        return self.dataset



if __name__ == "__main__":
    
    datasetdir = "./DataSets/GNN/"
    datasetpath = "./DataSets/GNN/ExchangeGNN.pth"


    stntypes = ['Uniaxial_X', 'Biaxial']
    strains = [np.linspace(-0.15, 0.15, 21), np.linspace(-0.12, 0.12, 15)]
    datagen = DataGenerator(rcut=5.0, stntypes=stntypes, strains=strains, phase='FM')

    dataset = datagen.generate(datasetdir=datasetdir)

    print(f"Generated dataset with {len(dataset)} samples. Saving to {datasetpath}...")
    torch.save(dataset, datasetpath)

