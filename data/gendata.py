import os
import torch
import numpy as np
from torch_geometric.data import Data
from ase.db import connect

from logger import getlogger

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

        self.lprefix = "[DataGenerator] "
        self.logger = getlogger()
    


    def parse(self, wkdir, stntype, stnvalue, rattleidx, fthresh=5.0):
        pwipath = os.path.join(wkdir, f"Strain_{stntype}_{stnvalue:.4f}_{rattleidx}", "espresso.pwi")
        pwopath = os.path.join(wkdir, f"Strain_{stntype}_{stnvalue:.4f}_{rattleidx}", "espresso.pwo")

        if not os.path.exists(pwipath):
            self.logger.warning(f"{self.lprefix}Input file not found for strain {stnvalue:.4f} at {pwipath}. Skipping...")
            return None, None

        if not os.path.exists(pwopath):
            self.logger.warning(f"{self.lprefix}Output file not found for strain {stnvalue:.4f} at {pwopath}. Skipping...")
            return None, None

        # atoms = self.crI3.strain_atoms(stntype=stntype, stnvalue=stnvalue)
        atoms = self.crI3.batoms.copy()
        atomsout = self.hubbard.parse(pwipath, pwopath, atoms)

        z = torch.tensor(atomsout.get_atomic_numbers(), dtype=torch.long)
        pos = torch.tensor(atomsout.get_positions(), dtype=torch.float32)
        energy = atomsout.get_potential_energy()
        forces = atomsout.get_forces()
        cell = torch.tensor(np.array(atomsout.get_cell()), dtype=torch.float32)

        if forces.any() > fthresh:
            self.logger.warning(f"{self.lprefix}Warning: Large forces detected for strain {stnvalue:.4f} (Rattle idx: {rattleidx}). Max force magnitude: {np.max(np.linalg.norm(forces, axis=1)):.4f} eV/Å. Skipping this sample.")
            return None, None

        #  Parse TB2J outputs for this strain
        # tb2jpath = os.path.join(wkdir, f"Strain_{stntype}_{stnvalue:.4f}", "tmp/TB2J_results/Multibinit", "exchange.xml")
        # if not os.path.exists(tb2jpath):
        #     print(f"TB2J output file not found for strain {stnvalue:.4f} at {tb2jpath}. Skipping TB2J parsing...")
            # self.logger.warning(f"{self.lprefix}TB2J output file not found for strain {stnvalue:.4f} at {tb2jpath}. Skipping TB2J parsing...")
        edgeidxs, edgeshifts = self.cgraph.tensorgraph(self.rcut)
        
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

        return data, atomsout
    
    def generate(self, datasetdir, rattleidxs):
        """
        Main method to generate the dataset. It iterates over all strains, parses the DFT and TB2J outputs, and saves the PyG Data objects.
        """
        db = connect('./DataSets/GNN/RattleGNN-Strain_0.0-Rcut_7.0.db')
        numsamples = 0
        for stntype, stnvalues in zip(self.stntypes, self.strains):
            wkdirs = [os.path.join(datasetdir, dataset) for dataset in [f"Rattle-{stntype}/{self.phase}", f"Rattle-{stntype}-4/{self.phase}"]]
            for wkdir in wkdirs:
                if wkdir.split('/')[-2] == 'Rattle-Uniaxial_X':
                    continue
                print(f"Processing Rattle: {wkdir.split('/')[-2]} for stntype: {stntype}...")
                for strain in stnvalues:
                    if strain != 0.0:
                        continue # Only use unstrained samples for baseline model

                    for rattleidx in rattleidxs:
                        print(f"Processing strain {strain:.4f} ({rattleidx+1}/{len(rattleidxs)})...")
                        
                        data, atoms = self.parse(wkdir, stntype, strain, rattleidx)
                        if data is not None:
                            self.dataset.append(data)
                            db.write(atoms)
                            numsamples += 1
                        else:
                            print(f"Data parsing failed for strain {strain:.4f} (Rattle idx: {rattleidx}). Skipping this sample.")
            
            # wkdir = 
            # for strain in stnvalues:
            #     for rattleidx in rattleidxs:
            #         print(f"Processing strain {strain:.4f} ({rattleidx+1}/{len(rattleidxs)})...")
                    
            #         data = self.parse(wkdir, stntype, strain, rattleidx)
            #         if data is not None:
            #             self.dataset.append(data)
            #             numsamples += 1
            #         else:
            #             print(f"Data parsing failed for strain {strain:.4f} (Rattle idx: {rattleidx}). Skipping this sample.")

        print(f"Dataset generation complete. Total samples: {numsamples}")

        return self.dataset



if __name__ == "__main__":
    from logging import getLogger
    logger = getLogger(__name__)

    RCUT = 7.0    
    datasetdir = "./DataSets/GNN/"
    datasetpath = f"./DataSets/GNN/RattleGNN-Strain_0.0-Rcut_{RCUT:.1f}.pth"

    stntypes = ['Biaxial', 'Uniaxial_X', 'Shear_XY']
    strains = [[float(s) for s in np.linspace(-0.12, 0.12, 15) if -0.05 < s < 0.05]]
    strains += [[float(s) for s in np.linspace(-0.15, 0.15, 21) if -0.05 < s < 0.05]]
    strains += [[float(s) for s in np.linspace(-0.15, 0.15, 21) if -0.05 < s < 0.05]]
    print('strains: ', strains)
    rattleidxs = list(range(10))
    datagen = DataGenerator(RCUT, stntypes=stntypes, strains=strains, phase='FM')

    dataset = datagen.generate(datasetdir=datasetdir, rattleidxs=rattleidxs)

    print(f"Generated dataset with {len(dataset)} samples. Saving to {datasetpath}...")
    torch.save(dataset, datasetpath)

