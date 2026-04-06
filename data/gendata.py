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
        
        edgeidxs, edgeshifts = self.cgraph.tensorgraph(self.rcut, atomsout)

        #  Parse TB2J outputs for this strain
        tb2jpath = os.path.join(wkdir, f"Strain_{stntype}_{stnvalue:.4f}_{rattleidx}", "tmp/TB2J_results/Multibinit", "exchange.xml")
        if not os.path.exists(tb2jpath):
            self.logger.warning(f"{self.lprefix}TB2J output file not found for strain {stnvalue:.4f} at {tb2jpath}. Skipping TB2J parsing...")
            cr_edges, exchangejs, cr_shifts, cr_edgedists = None, None, None, None
        else:
            cr_edges, exchangejs, cr_shifts, cr_edgedists = self.egraph.graph(tb2jpath, self.rcut, atomsout)

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
            cr_edge_dist=cr_edgedists,
            y_exchange=exchangejs,
            exchange = torch.tensor([True if exchangejs is not None else False], dtype=torch.bool)
        )

        return data, atomsout
    
    def generate(self, datasetdir, rattleidxs):
        """
        Main method to generate the dataset. It iterates over all strains, parses the DFT and TB2J outputs, and saves the PyG Data objects.
        """
        # db = connect('./DataSets/GNN/RattleGNN-Strain_0.0-Rcut_7.0.db')
        numsamples = 0
        for stntype, stnvalues in zip(self.stntypes, self.strains):
            # wkdirs = [os.path.join(datasetdir, dataset) for dataset in [f"Rattle-Exchange-{stntype}/{self.phase}", f"Rattle-Exchange-{stntype}-4/{self.phase}"]]
            wkdirs = [os.path.join(datasetdir, f"Rattle-Exchange-{stntype}-4/{self.phase}")]
            for wkdir in wkdirs:
                self.logger.info(f"Processing work directory: {wkdir.split('/')[-2]} for stntype: {stntype}...")
                for strain in stnvalues:
                    for rattleidx in rattleidxs:
                        self.logger.info(f"Processing strain (rattled) {strain:.4f} ({rattleidx+1}/{len(rattleidxs)})...")
                        
                        data, atoms = self.parse(wkdir, stntype, strain, rattleidx)
                        if data is not None:
                            self.dataset.append(data)
                            # db.write(atoms)
                            numsamples += 1
                        # else:
                            # self.logger.warning(f"Data parsing failed for strain {strain:.4f} (Rattle idx: {rattleidx}). Skipping this sample.")

        self.logger.info(f"Dataset generation complete. Total samples: {numsamples}")

        return self.dataset



if __name__ == "__main__":
    from pathlib import Path
    from logging import getLogger
    logger = getLogger(__name__)

    RCUT = 7.0    
    datasetdir = "./DataSets/MixedDataset/"
    datasetpath = f"./DataSets/GNN/MixedDataset_All-Rcut_{RCUT:.1f}.pth"

    # stntypes =  ['P-Exchange-Biaxial', 'P-Exchange-Shear_XY', 'P-Exchange-Uniaxial_X',]
    # stntypes += ['P-Rattle-Biaxial-2', 'P-Rattle-Shear_XY-2', 'P-Rattle-Uniaxial_X-2']
    # stntypes += ['P-Rattle-Biaxial-4', 'P-Rattle-Shear_XY-4', 'P-Rattle-Uniaxial_X-4']
    # stntypes += ['Rattle-Biaxial-2', 'Rattle-Shear_XY-2', 'Rattle-Uniaxial_X-2']
    # stntypes += ['Rattle-Biaxial-4', 'Rattle-Shear_XY-4', 'Rattle-Uniaxial_X-4', ]
    # stntypes += ['P-Rattle-Exchange-Uniaxial_X-4']

    # dirlist = os.listdir(datasetdir)
    dirpath = Path(datasetdir)
    dirlist = [d.name for d in dirpath.iterdir() if not d.name.startswith(".") and d.is_dir()]
    logger.info(f"Found directories: {dirlist}")
    stntypes = []
    strains = []
    numsamples = 0
    for dirname in dirlist:
        stntypes.append(dirname)

        stnvalues = []
        stndir = Path(os.path.join(datasetdir, dirname, "FM"))
        stnlist = [d.name for d in stndir.iterdir() if not d.name.startswith(".") and d.is_dir()]
        for stn in stnlist:
            stnvalues.append(float(stn.split('_')[-2]))
            numsamples += 1

        strains.append(stnvalues)

    logger.info(f"Number of strain types: {len(stntypes)}")
    logger.info(f"Total samples: {numsamples}")
    logger.info(f"Strains: {strains[0]}")
    
    rattleidxs = list(range(10))
    datagen = DataGenerator(RCUT, stntypes=stntypes, strains=strains, phase='FM')

    dataset = datagen.generate(datasetdir=datasetdir, rattleidxs=rattleidxs)

    datagen.logger.info(f"Saving to {datasetpath}...")
    torch.save(dataset, datasetpath)

